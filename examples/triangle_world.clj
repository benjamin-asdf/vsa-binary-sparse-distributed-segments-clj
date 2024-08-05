(ns triangle-world
  (:require
   [tech.v3.datatype.functional :as f]
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

;; - The triangle world is a finite state automaton with three states
;;   #{a, b, c}
;; - 2 input symbols
;;   #{:left, :right}
;;
;; - Transition function as schemata:
;;
;;  :c  <->  :a  <->  :b  <-> :c  <-> :a
;;
;;
(def world
  {[:c :right] :a
   [:a :right] :b
   [:b :right] :c
   [:a :left] :c
   [:b :left] :a
   [:c :left] :b})

;; -------------------------------

(defprotocol RandomAccessMemory
  (remember [this addr content])
  (recover [this addr-prime]
           [this addr-prime top-k]))

(def memory
  (let [sdm (sdm/->sdm {:address-count (long 1e5)
                        :address-density 0.00003
                        :word-length (long 1e4)})]
    (reify
      RandomAccessMemory
        (remember [this addr content]
          (sdm/write sdm addr content 1))
        (recover [this addr-prime top-k]
          ;; didn't figure out the interface yet
          ;; so I specify it in user space here.
          (some->
            (sdm/converged-lookup-impl
              sdm
              addr-prime
              {:decoder-threshold 1
               :stop? (fn [acc next-outcome]
                        (cond
                          (< 0.9 (:confidence next-outcome))
                            {:stop-reason :high-confidence
                             :success? true}
                          ;;
                          ;; two times low
                          ;; confidence, means
                          ;; it's diverging I
                          ;; think
                          ;;
                          (when-let [last-confidence
                                       (:confidence
                                         (peek (:result-xs
                                                 acc)))]
                            (< (+ (:confidence next-outcome)
                                  last-confidence)
                               0.2))
                            {:stop-reason :diverging
                             :success? false}))
               :top-k top-k})
            :result-address
            sdm/torch->jvm
            (dtt/->tensor :datatype :int8)))
      (recover [this addr-prime]
        (recover this addr-prime 1)))))

;; -------------------------------
;; cognitive architecture
;;
;;
;;
;;
;;           Explore mode:
;;
;;           t1:
;;           - world state s1 -> sensor register, called "source"
;;           - action (random play) a1 -> action register
;;           - focus = source ⊙ action , s1 ⊙ a1
;;           - focus is the address for memory <- this is delayed by 1 time step (for instance with memory delay lines)
;;
;;           t2:
;;           - world state s2 is 'sensed', immediately permuted and put into the input register, "destination": p(s2)
;;           - action register (random play), a2
;;           - focus = source ⊙ action, s2 ⊙ a2
;;           - update memory, in effect hetero associate:
;;
;;             s1 ⊙ a1 -> p(s2)
;;
;;           memory address  -> memory content
;;
;;
;;
;;
;;
;;            world                 sensor register
;;           +---------+           +----------+
;;           |         |           |          |                2                   permute  = 'destination'
;;           |    s    |<----------+          +-----------------------------------------|
;;           |         |           |          |                                         |
;;           +---------+           +-----+----+                                  +------v-----+
;;               ^                       |                                       |            | input register
;;               |                       |                                       |            |
;;               |                       | 'source'                              +------+-----+
;;               |                       |                                              |
;;               |                       |                     +------+          +------v-----+
;;               |                       |                     |      |          |            |
;;               |                       |                     |      |          |            |
;;               |                       |   1   ⊙            |      |   2      |            |
;;             +-|------+                +-------+--------->   |      +------->  |            |
;;             |        |                        |             |      |  addr    |            |
;;             |        -------------------------+             |      |   ^      |            |
;;             |        |               1                      |      |   |      |            |
;;             +--------+                                      +------+   |      +------------+
;;             action register          'action'                          |
;;                                                             focus      |            memory
;;                                                                        |
;;                                                                        |
;;                                                                        |
;;                                                                      delay by 1
;;
;;
;; 1 active at t - 1
;; 2 active at t
;;
;; -----------------------------------------------------------------
;;
;;
;; Usage modes:
;;
;;   Use the memory
;;   approaching with
;;   sensor state ⊙ action results in a 'predicted' destination
;;
;; -----------------------------------------------------------------

;;
;;
;; cognitive architecture - implementation
;;
;; -----------------------------------------------------------------
;;
;; calling it 'destination-rememberer'


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

;; -------------------------------
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

(reductions
  (fn [{:keys [cog-state world-state]} n]
    (let [cog-state (destination-rememberer-update
                      cog-state
                      (hdd/clj->vsa world-state))
          action (cog-state->action cog-state)
          new-world (update-world world-state action)]
      {:action action
       :cog-state cog-state
       :n n
       :state-action-outcome [world-state action new-world]
       :world-state new-world}))
  {:action nil
   :cog-state
     (destination-rememberer-state
       (let [actions (into [] (hdd/clj->vsa* actions))]
         (fn [] (rand-nth actions)))
       (fn [addr content] (remember memory addr content)))
   :world-state :a}
  (range 10))

(hdd/cleanup*
 (hd/permute-inverse (recover memory (hdd/clj->vsa* {:a :left}))))
'(:c)

(for [state [:a :b :c]
      action [:left :right]]
  (let [prediction (hd/permute-inverse
                    (recover memory
                             (hdd/clj->vsa* {action
                                             state})))]
    [:real [state action (world [state action])] :cog
     [state action (hdd/cleanup prediction)] :success?
     (= (world [state action]) (hdd/cleanup prediction))]))

'([:real [:a :left :c] :cog [:a :left :c] :success? true]
 [:real [:a :right :b] :cog [:a :right :b] :success? true]
 [:real [:b :left :a] :cog [:b :left :a] :success? true]
 [:real [:b :right :c] :cog [:b :right :c] :success? true]
 [:real [:c :left :b] :cog [:c :left :b] :success? true]
 [:real [:c :right :a] :cog [:c :right :a] :success? true])


;; ------------------------------------
;; Version II:
;;
;; rembers both ways
;; --------------
;;
;; - The same but we update the memory with both the focus and the content.
;; - note that by permuting, we effectively say 'caused by me', separating mere world states from
;;   action outcomes.
;;
;;
;; Usage:
;; ----------------
;;
;; - Note that each destination has 2 possible sources
;; - i.e. `:c` can be reached from both `:a` and `:b`, with `:left` and `:right` actions respectively.
;; - Thus, querying the memory with p(:c) and top-k = 1 results roughly in a coin flip between
;;   [:a :left] and [:b :right]
;; - By querying with top-k = 2, we find the superposition of 'possible paths'
;;


(defn destination-rememberer-update-2
  [{:as state
    :keys [t action-register sensor-register focus
           memory-remember play-state]} next-world-state]
  ;; -------------------------------------------------
  ;; focus -> destination
  (memory-remember focus (hd/permute next-world-state))
  ;; destination -> focus
  (memory-remember (hd/permute next-world-state) focus)
  ;; -------------------------------------------------
  (let [new-action (play-state)
        new-focus (hd/bind new-action next-world-state)]
    (-> state
        (update :t inc)
        (assoc :focus new-focus)
        (assoc :sensor-register next-world-state)
        (assoc :action-register new-action))))

(time
 (do
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
       (fn [addr content]
         (remember memory addr content)))
      :world-state :a}
     (range 20)))
   nil))
;; "Elapsed time: 113.821859 msecs"


(for [state [:a :b :c]
      action [:left :right]
      ;; 'focus' = sources ⊙ actions
      ;; (possible paths to destination)
      :let [world-destination (world [state action])
            recoved-foci
            ;; query destination
            (recover memory
                     (hd/permute (hdd/clj->vsa
                                  world-destination))
                     2)]]
  [(= (world [state action])
      ;; 'prediction' like up top
      (some->> (recover memory
                        (hdd/clj->vsa* {action state}))
               (hd/permute-inverse)
               hdd/cleanup))
   (= action
      (some-> recoved-foci
              ;; unbind with the query state will
              ;; 'collapse' the outcome into one of the
              ;; action ⊙ state domains.
              (hd/unbind (hdd/clj->vsa* state))
              hdd/cleanup))])

'([true true]
  [true true]
  [true true]
  [true true]
  [true true]
  [true true])


(for [world-destination [:a :b :c]
      :let [recoved-foci
            ;; query destination
            (recover memory
                     (hd/permute (hdd/clj->vsa
                                  world-destination))
                     2)]]
  ;; but interesting, the system represents the notion
  ;; that a destination is reached in multiple ways,
  ;; by simply returning the superposition of foci
  ;; (action ⊙ state) bound-pairs.
  [:destination world-destination :how-to-get-there
   [:left
    (hdd/cleanup (hd/unbind recoved-foci
                            (hdd/clj->vsa* :left)))]
   [:right
    (hdd/cleanup (hd/unbind recoved-foci
                            (hdd/clj->vsa* :right)))]])



'([:destination :a :how-to-get-there [:left :b] [:right :c]]
   [:destination :b :how-to-get-there [:left :c] [:right :a]]
   [:destination :c :how-to-get-there [:left :a] [:right :b]])

;; ------------------
;;
;; - The downside here is that the user needs to know either all actions or world states
;; - No information about the action is recovered when one has a focus at hand.
;;   (because bind doesn't preserve similarity, one always needs a key to find a value).
;; - Braitenberg was imagining the Mnemotrix system, and this would help here.
;; - If the outcomes of actions are *also* associated with the actions, then we can recover the info
;;   from the memory
;; - other alternatives are having an action item memory around.
;; - perhaps a powerful auto associative memory that discretices actions, one can see that less
;;   degrees of freedom make sense in certain aspects of such a system.
;;   (I think this might have been counterintuitive).
;; - where you want complete openness is presumably in the sensory states
;;

;; ----------------------------------------
;; Use as non-deterministic finite state automaton
;;
;; Basically for the same reasons as why non-deterministic finite state automatons work.
;; See lit.org [7]
;;
;; This showcases how SDM happily responds with a mix of information, if approached with a mix of addresses.
;;

(let [destinations
        (recover memory
                 (hd/bind (hdd/clj->vsa* #{:a})
                          ;; corresponds to a
                          ;; 'generalized
                          ;; state' of a
                          ;; non-deterministic
                          ;; finite state automaton
                          (hdd/clj->vsa* #{:right :left}))
                 2)]
  (hdd/cleanup* (hd/permute-inverse destinations)))
'(:c :b)

;; one can get wilder, in effect querying with the crossproduct of
;; actions ⊙ states
;;
;; note that one needs to increase top-k accordingly.
;; Roughly, 1 top-k corresponds to getting a seeds vector worth of data out of the SDM,
;; if we query for superpositions, increase it.
;;
(let [destinations
      (recover memory
               (hd/bind (hdd/clj->vsa* #{:a :b})
                        ;; corresponds to a
                        ;; 'generalized
                        ;; state' of a
                        ;; non-deterministic
                        ;; finite state automaton
                        (hdd/clj->vsa* #{:right :left}))
               4)]
  (hdd/cleanup* (hd/permute-inverse destinations)))
'(:c :a :b)


;; ----------------------------------------
;; Basic analogy fabric
;;
;; - this is very simplistic, but that is often the point in software engineering,
;;   being able to keep the overview, until the system is sophisticated, yet it's simplicity and
;;   the overview stays.
;;
;;












;; ----------------------------------------
;; Discussion
;;
;; -
