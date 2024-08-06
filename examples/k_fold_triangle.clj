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

;; (defprotocol RandomAccessMemory
;;   (remember [this addr content])
;;   (recover [this addr-prime]
;;     [this addr-prime top-k]))

;; (defn ->memory
;;   []
;;   ;; let's start with delays = 2
;;   (let [sdm (sdm/k-fold-sdm {:address-count (long 1e5)
;;                              :address-density 0.00003
;;                              :k-delays 2
;;                              :word-length (long 1e4)})]
;;     (reify
;;       RandomAccessMemory
;;         (remember [this addr content]
;;           (sdm/write sdm addr content 1))
;;         (recover [this addr-prime top-k]
;;           (sdm/lookup sdm addr-prime top-k 1)
;;           ;; (let [lookup-result]
;;           ;;   (some-> lookup-result
;;           ;;           (dtt/->tensor :datatype :int8)))
;;         )
;;         (recover [this addr-prime]
;;           (recover this addr-prime 1)))))

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
  (let [lookup-outcome (recover-1 sdm addr-prime top-k)]))

(let [m (->memory)]
  (remember m (hdd/clj->vsa* :a) (hdd/clj->vsa* :x))
  (remember m (hdd/clj->vsa* :x) (hdd/clj->vsa* :b))
  [(hdd/cleanup* (some-> (recover-1 m (hdd/clj->vsa* :a) 1)
                         :result
                         sdm/torch->jvm
                         (dtt/->tensor :datatype :int8)))
   (hdd/cleanup* (some-> (recover-1 m (hd/->empty) 1)
                         :result
                         sdm/torch->jvm
                         (dtt/->tensor :datatype :int8)))])
'[(:x) (:b)]

(let [m (->memory)]
  (remember m (hdd/clj->vsa* :a) (hdd/clj->vsa* :x))
  (remember m (hdd/clj->vsa* :x) (hdd/clj->vsa* :b))
  (time (doall (for [n (range 1e3)]
                 (remember m (hd/->seed) (hd/->seed)))))
  [(recover-1 m (hd/drop (hdd/clj->vsa* :a) 0.5) 1)])



[{([0. 0. 0. ... 0. 0. 0.] device='cuda:0')
  :address-location-count 46
  :confidence 0.3532608449459076
  :result tensor}]


[{([0. 0. 0. ... 0. 0. 0.] device='cuda:0')
  :address-location-count 35
  :confidence 0.37714284658432007
  :result tensor}]

























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
   ;; intialize with non-sense
   :action-register (hd/->seed)
   :sensor-register (hd/->seed)
   :focus (hd/->seed)})

(defn destination-rememberer-update
  [{:as state :keys [focus memory-remember play-state]} next-world-state]
  (memory-remember focus (hd/permute next-world-state))
  (let [new-action (play-state)
        new-focus (hd/bind new-action next-world-state)]
    (-> state
        (update :t inc)
        (assoc :focus new-focus)
        (assoc :sensor-register next-world-state)
        (assoc :action-register new-action))))
