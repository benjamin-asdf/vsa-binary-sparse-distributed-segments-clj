(ns k-fold-triangle
  (:require [tech.v3.datatype.functional :as f]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
            [tech.v3.datatype.bitmap :as bitmap]
            [fastmath.random :as fm.rand]
            [fastmath.core :as fm]
            [bennischwerdtner.sdm.sdm :as sdm]
            [bennischwerdtner.hd.binary-sparse-segmented :as
             hd]
            [bennischwerdtner.pyutils :as pyutils]
            [tech.v3.datatype.unary-pred :as unary-pred]
            [tech.v3.datatype.argops :as dtype-argops]
            [bennischwerdtner.hd.data :as hdd]))


;; -----------------------------------
;; See triangle_world.clj.
;; (this is V2)
;;

;; -----------------------------------
;; - Use a similar architecture, but k-fold memory
;; - This allows the model to learn sequences
;;

(def world
  {[:c :right] :a
   [:a :right] :b
   [:b :right] :c
   [:a :left] :c
   [:b :left] :a
   [:c :left] :b})

;; -------------------------------------

(defn ->memory
  []
  ;; let's start with delays = 2
  (sdm/k-fold-sdm {:address-count (long 1e5)
                   :address-density 0.00003
                   :k-delays 2
                   :word-length (long 1e4)}))

(defn remember [sdm addr content]
  (sdm/write sdm addr content 1))

(defn recover-1
  [sdm addr-prime top-k]
  (sdm/lookup sdm addr-prime top-k 1))

(defn recover
  [sdm addr-prime top-k]
  (let [lookup-outcome (recover-1 sdm addr-prime top-k)]
    (when (< 0.1 (:confidence lookup-outcome))
      (some-> lookup-outcome
              :result
              sdm/torch->jvm
              (dtt/->tensor :datatype :int8)))))

(let [m (->memory)]
  (remember m (hdd/clj->vsa* :a) (hdd/clj->vsa* :x))
  (remember m (hdd/clj->vsa* :x) (hdd/clj->vsa* :b))
  [(hdd/cleanup* (some-> (recover-1 m (hdd/clj->vsa* :a) 1)
                         :result
                         sdm/torch->jvm
                         (dtt/->tensor :datatype :int8)))
   ;; query with empty to find 'next'
   (hdd/cleanup* (some-> (recover-1 m (hd/->empty) 1)
                         :result
                         sdm/torch->jvm
                         (dtt/->tensor :datatype :int8)))])
'[(:x) (:b)]



;; [{([0. 0. 0. ... 0. 0. 0.] device='cuda:0')
;;   :address-location-count 35
;;   :confidence 0.37714284658432007
;;   :result tensor}]

























;; -----------------------------------
;; Cogntive architecture:
;;
;; The same as triangle_world.clj
;; But we don't need to delay or keep history.
;; Instead use a k-fold sdm.
;;
;;
;;
;;  focus:
;;  action ⊙ sensor-data
;;  -> use focus as address for memor
;;
;;  At the same time step, update sdm:
;;  p(sensor-data) -> content
;;
;;
;;
;; t1:
;;
;; #{ a ⊙ right } ->  p(a)
;;
;; t2:
;;
;; #{ b ⊙ right , delay-addresses[a ⊙ right, 1] }  ->  p(b)
;;
;; -----------------------------------
;;
;; Usage mode:
;;
;; - Query with a source, step sdm time 1 once, then read the prediction
;;
;;

(defn destination-rememberer-state
  [play-state memory-remember]
  {:t 0
   :memory-remember memory-remember
   :play-state play-state
   ;; doesn't need to keep history
   ;; intialize with non-sense
   :action-register (hd/->seed)
   ;; :sensor-register (hd/->seed)
   ;; :focus (hd/->seed)
  })

(defn destination-rememberer-update
  [{:as state :keys [memory-remember play-state]}
   next-world-state]
  (let [new-action (play-state)
        new-focus (hd/bind new-action next-world-state)]
    (memory-remember new-focus
                     (hd/permute next-world-state))
    (-> state
        (update :t inc)
        ;; (assoc :focus new-focus)
        ;; (assoc :sensor-register next-world-state)
        (assoc :action-register new-action))))

;; -----------------------------------------------------
;; effectors

(def actions-item-memory
  (hdd/->TinyItemMemory
   (atom {:left (hdd/clj->vsa :left) :right (hdd/clj->vsa :right)})))

(def cleanup-action #(hdd/m-cleanup actions-item-memory %))

(def actions [:left :right])

(def cog-state->action (comp cleanup-action :action-register))

;; ---------------------------------
;; world

(defn update-world [state action]
  (world [state action] state))

;; -------------------------------
;; training via self-play
;;

(def outcome
  (let [m (->memory)]
    [m
     (doall
       (reductions
         (fn [{:keys [cog-state world-state]} n]
           (let [cog-state (destination-rememberer-update
                             cog-state
                             (hdd/clj->vsa world-state))
                 action (cog-state->action cog-state)
                 new-world (update-world world-state
                                         action)]
             {:action action
              :cog-state cog-state
              :n n
              :state-action-outcome [world-state action
                                     new-world]
              :world-state new-world}))
         {:action nil
          :cog-state
            (destination-rememberer-state
              (let [actions (into []
                                  (hdd/clj->vsa* actions))]
                (fn [] (rand-nth actions)))
              (fn [addr content] (remember m addr content)))
          :world-state :a}
         (range 20)))]))

(let [[m lst] outcome]
  lst
  (recover-1 m (hdd/clj->vsa* {:a :right}) 1)
  (hdd/cleanup*
    (hd/permute-inverse
      (recover m (hdd/clj->vsa* {:a :right}) 1)))
  (for [state [:a :b :c]
        action [:left :right]]
    (let [prediction
            (hd/permute-inverse
              (do
                (recover m (hdd/clj->vsa* {action state}) 1)
                ;; double req to get to the prediction
                (recover m (hd/->empty) 1)))]
      [:real [state action (world [state action])] :cog
       [state action (hdd/cleanup prediction)] :success?
       (= (world [state action])
          (hdd/cleanup prediction))])))

'([:real [:a :left :c] :cog [:a :left :c] :success? true]
  [:real [:a :right :b] :cog [:a :right :b] :success? true]
  [:real [:b :left :a] :cog [:b :left :a] :success? true]
  [:real [:b :right :c] :cog [:b :right :c] :success? true]
  [:real [:c :left :b] :cog [:c :left :b] :success? true]
  [:real [:c :right :a] :cog [:c :right :a] :success? true])

;; genaralized state works the same:
(let [[m lst] outcome]
  (let [state #{:a :b}
        action :right
        prediction
        (hd/permute-inverse
         (do
           ;; top-k only has an effect for the recover, you could even say 0
           (recover m (hdd/clj->vsa* {action state}) 0)
           ;; double req to get to the prediction
           (recover m (hd/->empty) 2)))]
    [state action (hdd/cleanup* prediction)]))
'[#{:b :a} :right (:c :b)]


;; -----------------------------------
;; Comparator V1
;; -----------------------------------
;; Now you can sort of 'roll' with the world, and say whether it is what you expect.
;;
































(comment
  (let [m (->memory)]
    (remember m (hdd/clj->vsa* :a) (hdd/clj->vsa* :x))
    (remember m (hdd/clj->vsa* :x) (hdd/clj->vsa* :b))
    (time (doall (for [n (range 1e3)]
                   (remember m (hd/->seed) (hd/->seed)))))
    [(recover-1 m (hd/drop (hdd/clj->vsa* :a) 0.5) 1)])
  [{([0. 0. 0. ... 0. 0. 0.] device='cuda:0')
    :address-location-count
    0.3532608449459076 :result
    46 :confidence
    tensor}]

  (let [m (->memory)]
    (remember m (hdd/clj->vsa* :a) (hdd/clj->vsa* :x))
    (remember m (hdd/clj->vsa* :x) (hdd/clj->vsa* :b))
    [(recover-1 m (hd/drop (hdd/clj->vsa* :a) 0.8) 1)])

  ;; [
  ;;  {([0. 0. 0. ... 0. 0. 0.] device='cuda:0')
  ;;   :address-location-count
  ;;   0.11428571492433548 :result
  ;;   35 :confidence
  ;;   tensor}]

  (hdd/cleanup* (-> *1
                    peek
                    :result
                    sdm/torch->jvm
                    (dtt/->tensor :datatype :int8)))

  (:x)

  (let [m (->memory)]
    (remember m (hdd/clj->vsa* :a) (hdd/clj->vsa* :x))
    (remember m (hdd/clj->vsa* :x) (hdd/clj->vsa* :b))
    [(recover-1 m (hd/drop (hdd/clj->vsa* :a) 0.9) 1)])
  ;; [{:address-location-count 34, :confidence 0.029411764815449715, :result tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0')}]

  (hdd/cleanup* (-> *1
                    peek
                    :result
                    sdm/torch->jvm
                    (dtt/->tensor :datatype :int8)))
  (:x))




;; -----------------------------------

;; -----------------------------------
;; Since time is flowing automatically, it's harder to easily put the reverse
;;
;;
;;

;; - you could rember both directions at once like
'(memory-remember
  (hd/superposition new-focus (hd/permute next-world-state))
  (hd/superposition new-focus (hd/permute next-world-state)))

;;
;; But then there is no direction anymore?
;;
(defn destination-rememberer-update-2
  [{:as state :keys [memory-remember play-state]} next-world-state]
  (let [new-action (play-state)
        new-focus (hd/bind new-action next-world-state)]
    (memory-remember
     (hd/superposition new-focus (hd/permute next-world-state))
     (hd/superposition new-focus (hd/permute next-world-state)))
    (-> state
        (update :t inc)
        ;; (assoc :focus new-focus)
        ;; (assoc :sensor-register next-world-state)
        (assoc :action-register new-action))))

(def outcome
  (let [m (->memory)]
    [m
     (doall
      (reductions
       (fn [{:keys [cog-state world-state]} n]
         (let [cog-state (destination-rememberer-update-2
                          cog-state
                          (hdd/clj->vsa world-state))
               action (cog-state->action cog-state)
               new-world (update-world world-state action)]
           {:action action
            :cog-state cog-state
            :n n
            :state-action-outcome [world-state action
                                   new-world]
            :world-state new-world}))
       {:action nil
        :cog-state
        (destination-rememberer-state
         (let [actions (into [] (hdd/clj->vsa* actions))]
           (fn [] (rand-nth actions)))
         (fn [addr content] (remember m addr content)))
        :world-state :a}
       (range 50)))]))

(let [[m lst] outcome]
  (recover-1 m (hdd/clj->vsa* {:a :right}) 2)
  (some->
   (recover-1 m (hd/->empty) 2)
   :result
   sdm/torch->jvm
   (dtt/->tensor :datatype :int8)
   (hd/permute-inverse)
   (hdd/cleanup*)))

;; Hm, I tried this but this didn't have good results.
(let [[m lst] outcome]
  (let [r (some-> (recover-1 m
                             (hd/superposition
                               (hdd/clj->vsa* {:c :left})
                               (hd/permute (hdd/clj->vsa*
                                             :b)))
                             3)
                  :result
                  sdm/torch->jvm
                  (dtt/->tensor :datatype :int8))]
    [:destination :b :how-to-get-there
     [:right
      (some-> r
              (hd/unbind (hdd/clj->vsa* :right))
              (hdd/cleanup*)) :left]]
    ;; doing something wrong ig.
    (hd/similarity (hdd/clj->vsa* :c)
                   (hd/unbind r (hdd/clj->vsa* :left)))))
