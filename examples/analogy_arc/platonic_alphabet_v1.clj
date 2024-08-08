(ns
    platonic-alphabet-v1
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

;; ----------------------------------------
;; platonic-alphabet is similar to triangle world, but more elements and not circular.
;;
;;

(def alphabet (into [] (map (comp keyword str char) (range (int \a) (inc (int \z))))))

(def world
  (into {}
        (concat (map (fn [a b] [[a :right] b])
                  alphabet
                  (drop 1 alphabet))
                (map (fn [a b] [[a :left] b])
                  (reverse alphabet)
                  (drop 1 (reverse alphabet))))))

;;
;; a right -> b
;; a left -> 'nothing' (but 'a' because of update-world implementation)

;; -----------------------------------------------
;; memory

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

;; ---------------------------------------
;; explorer system
;; (this is k-fold destination-rememberer)
;; (see k_fold_triangle.clj)
;;

(defn destination-rememberer-state
  [play-state memory-remember]
  {:action-register (hd/->seed)
   :memory-remember memory-remember
   :play-state play-state
   :t 0})

(defn destination-rememberer-update
  [{:as state :keys [memory-remember play-state]}
   next-world-state]
  (let [new-action (play-state)
        new-focus (hd/bind new-action next-world-state)]
    (memory-remember new-focus
                     (hd/permute next-world-state))
    (-> state
        (update :t inc)
        (assoc :action-register new-action))))


;; -------------------------------
;; training via self-play
;;
;;

(def outcome
  (let [m (->memory)]
    [m
     (time
      (doall
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
          (let
              [actions-xs
               (atom
                (cycle
                 (concat
                  (repeat 28 :right)
                  (repeat 28 :left))))]
              (fn []
                (hdd/clj->vsa* (first (swap! actions-xs next)))))
          (fn [addr content] (remember m addr content)))
         :world-state :a}
        (range 100))))]))

;; generalized state:
(let [[m lst] outcome]
  (let [state #{:a :b}
        action :right
        prediction
          (hd/permute-inverse
            (do
              ;; top-k only has an effect for the
              ;; recover, you could even say 0
              (recover m (hdd/clj->vsa* {action state}) 0)
              ;; double req to get to the prediction
              (recover m (hd/->empty) 2)))]
    [state action (hdd/cleanup* prediction)]))
'[#{:b :a} :right (:c :b)]

(let [[m lst] outcome]
  (let [state #{:x :a}
        action :right
        prediction
          (hd/permute-inverse
            (do
              ;; top-k only has an effect for the
              ;; recover, you could even say 0
              (recover m (hdd/clj->vsa* {action state}) 0)
              ;; double req to get to the prediction
              (recover m (hd/->empty) 2)))]
    [state action (hdd/cleanup* prediction)]))
'[#{:x :a} :right (:y :b)]

(let [[m lst] outcome]
  (let [state :x
        action :right
        prediction
        (hd/permute-inverse
         (do
           ;; top-k only has an effect for the
           ;; recover, you could even say 0
           (recover m (hdd/clj->vsa* {action state}) 0)
           ;; double req to get to the prediction
           (recover m (hd/->empty) 1)))]
    [state action (hdd/cleanup* prediction)]))

;; with top-k = 2, looks noisy
'[:x :right (:y :m :o :l :n)]
'[:x :right (:y)]

;; ------------------------
;;
;; now with comparator:
;;
;;

(defn comparator-state-2
  [predictor comperator]
  {:comperator comperator :predictor predictor :t 0})

(defn update-comparator-2
  [{:as state :keys [predictor comperator]} s-world]
  (let [p (predictor s-world)]
    (-> state
        (update :t inc)
        ;; diagnostic
        (assoc :prediction p)
        (assoc :comperator-output (comperator p s-world)))))


;; ----------------------------


(def world-seq (subvec alphabet 12 18))
(def action-seq (repeat :right))

(let [[m lst] outcome
      action :right
      ;; reset the memory
      _ (do (recover m (hd/->empty) 0)
            (recover m (hd/->empty) 0)
            (recover m (hd/->empty) 0))]
  (->> (reductions
         update-comparator-2
         (comparator-state-2
           (fn [s]
             (some->
               (recover m (hdd/clj->vsa* {action s}) 1)
               (hd/permute-inverse)))
           (fn [prediction s-world]
             (when prediction
               [(hdd/cleanup* prediction) s-world
                (hd/similarity prediction
                               (hdd/clj->vsa* s-world))])))
         world-seq)
       (drop 1)
       (map :comperator-output)))

'([(:m) :m 1.0]
 [(:n) :n 1.0]
 [(:o) :o 1.0]
 [(:p) :p 1.0]
 [(:q) :q 1.0]
  [(:r) :r 1.0])

;; ... comparator says 'yes' to the sequence


;; perturbing it:

(let [[m lst] outcome
      action :right
      _ (do (recover m (hd/->empty) 0)
            (recover m (hd/->empty) 0)
            (recover m (hd/->empty) 0))]
  (->> (reductions
        update-comparator-2
        (comparator-state-2
         (fn [s]
           (some->
            (recover m (hdd/clj->vsa* {action s}) 1)
            (hd/permute-inverse)))
         (fn [prediction s-world]
           (when prediction
             [(hdd/cleanup* prediction)
              s-world
              (hd/similarity prediction (hdd/clj->vsa* s-world))])))
        ;; not alphabet
        [:m :n :j :f :a])
       (drop 1)
       (map :comperator-output)))

'([(:m) :m 1.0]
  [(:n) :n 1.0]
  ;; predictect o, got j
  [(:o) :j 0.0]
  [(:f) :f 1.0]
  [(:g) :a 0.0])


;; ->
;; kinda cool, a seq that doesn't fit is 'confusing'
;; i.e 0.0 could wire 'no' line for the comparator,
;; signaling novelity or surprise.
;;


;; experiment, query with generilized state:
(let [[m lst] outcome
      action :right]
  (->> (reductions
        update-comparator-2
        (comparator-state-2
         (fn [s]
           (some->
            (recover m (hdd/clj->vsa* {action s}) 1)
            (hd/permute-inverse)))
         (fn [prediction s-world]
           (when prediction
             [(hdd/cleanup* prediction) s-world
              (hd/similarity prediction
                             (hdd/clj->vsa* s-world))])))
        [ ;; first state is a generalized state,
         ;; complete alphabet
         (hdd/clj->vsa* (into #{} alphabet)) :j :k :l])
       (drop 1)
       (map :comperator-output)))

;; not sure what I expected

;; '([(:m) #tech.v3.tensor<int8> [10000] [0 0 0 ... 0 0 0] 1.0]
;;   [(:m :n :z :l :x) :j 0.05]
;;   [(:k) :k 1.0]
;;   [(:l) :l 1.0])

;; --------------------
;; Discussion:
;;
;; - The outcome of this is the set of 'most relevant' symbols in the sdm.
;; - It's simply the most extreme generilized state, given the alphabet.
;;

(float (/ (.indexOf alphabet :m) (count alphabet)))
0.46153846

;; :m is right at the center of the alphabet.
;; Actually makes sense that via the left right self play, it was visited most often.
;;



;; ------------------------------------------
;; Substring search
;;
;; - this is for trying out in what ways the trainend alphabet model can be used.
;;

;; either way, we can say something is a substring by looking at the prediction confidence

(defn substring-confidence
  [m substring-symbols]
  (let [action :right
        _ (do (recover m (hd/->empty) 0)
              (recover m (hd/->empty) 0)
              (recover m (hd/->empty) 0))]
    (f//
     (f/sum
      (->> (reductions
            update-comparator-2
            (comparator-state-2
             (fn [s]
               (some-> (recover m
                                (hdd/clj->vsa*
                                 {action s})
                                1)
                       (hd/permute-inverse)))
             (fn [prediction s-world]
               (when prediction
                 [(hdd/cleanup* prediction) s-world
                  (hd/similarity prediction (hdd/clj->vsa* s-world))
                  ])))
            substring-symbols)
           (drop 1)
           (map :comperator-output)
           (map #(nth % 2))))
     (count substring-symbols))))

;; trying some random alphabetic substrings:

(for [n (range 20)]
  (let
      [n (rand-int (inc (count alphabet)))
       length (inc (rand-int 6))
       rand-substr (subvec alphabet n (min (+ n length) (count alphabet)))]
      (when (seq rand-substr)
        (substring-confidence (first outcome) rand-substr))))
'(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 nil 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0)


;; trying to perturb it:

(for [n (range 20)]
  (let
      [n (rand-int (inc (count alphabet)))
       length (inc (rand-int 6))
       rand-substr (subvec alphabet n (min (+ n length) (count alphabet)))
       ;; swap a random symbol for something else in the alphabet
       rand-substr (assoc
                    rand-substr
                    (rand-int (count rand-substr))
                    (rand-nth alphabet))]
    (when (seq rand-substr)
      [rand-substr
       (substring-confidence (first outcome) rand-substr)])))

'([[:c :c :d] 1.0]
  [[:s :v :u] 1.0]
  [[:v :w :x :y :e] 0.8]
  [[:s :b :u :v :w :x] 0.8833333333333333]
  [[:y :f] 1.0]
  [[:k] 1.0]
  [[:u :v :h :x :y] 0.8]
  [[:t :u :z :w :x] 0.8099999999999999]
  [[:c :f :e :f] 0.75]
  [[:w :x :f :z] 1.0]
  [[:l :g :n :o] 1.0]
  [[:w] 1.0]
  [[:e :k :g] 1.0]
  [[:q :r :l :t] 1.0]
  [[:k :v :m] 0.6666666666666666]
  [[:u :v :w :m :y] 0.8]
  [[:j :y] 1.0]
  [[:j :k :l :c :n] 0.8]
  [[:k :l :q :n :o] 0.6]
  [[:l] 1.0])

;;
;; Says 'yes' to stuff, because the incoming world data has a lot of effect
;;
;;
;; ------------------------------------------------------------------
;; (just) Experiments:
;;
;; (lack of narrative might be confusing, you can skip to 'use comparator v1')

(defn substring-confidence-2
  [m substring-symbols]
  (let [action :right
        _ (do (recover m (hd/->empty) 0)
              (recover m (hd/->empty) 0)
              (recover m (hd/->empty) 0))]
    (->> (reductions
          update-comparator-2
          (comparator-state-2
           (fn [s]
             (some-> (recover m
                              (hd/drop
                               (hdd/clj->vsa* {action s})
                               ;; it still
                               ;; resolves
                               ;; stuff even
                               ;; when 75% of
                               ;; bits are
                               ;; dropped
                               0.75)
                              1)
                     (hd/permute-inverse)))
           (fn [prediction s-world]
             (when prediction
               [(hdd/cleanup* prediction) s-world
                (hd/similarity prediction
                               (hdd/clj->vsa*
                                s-world))])))
          substring-symbols)
         (drop 1)
         (map :comperator-output)
         ;; (map #(nth % 2))
         )))

(for [n (range 5)]
  (let [n (rand-int (inc (count alphabet)))
        length (inc (rand-int 6))
        rand-substr (subvec alphabet
                            n
                            (min (+ n length)
                                 (count alphabet)))
        ;; swap a random symbol for something else in
        ;; the alphabet
        rand-substr (assoc rand-substr
                      (rand-int (count rand-substr))
                        (rand-nth alphabet))]
    (when (seq rand-substr)
      [rand-substr
       (substring-confidence-2 (first outcome)
                               rand-substr)])))

;;
;; I get it:
;; The current symbol has so much influence on the outcome
;;

;; lol, dropping from the 'world' when going through the sequence
;;
(for [n (range 20)
      :let [n (rand-int (inc (count alphabet)))
            length (inc (rand-int 6))
            rand-substr (subvec alphabet
                                n
                                (min (+ n length)
                                     (count alphabet)))]
      :when rand-substr]
  [rand-substr
   (f/mean (->> (substring-confidence-2 (first outcome)
                                        rand-substr)
                (map #(nth % 2))))])

'([[:g :h :i :j :k] 1.0]
 [[:q :r] 1.0]
 [[:x :y :z] 1.0]
 [[:c :d :e :f :g :h] 1.0]
 [[:m :n :o] 1.0]
 [[:a] 1.0]
 [[:i :j :k :l :m] 1.0]
 [[:g :h :i :j :k :l] 1.0]
 [[:l] 1.0]
 [[:y :z] 1.0]
 [[:v :w :x :y :z] 1.0]
 [[:q :r :s :t] 1.0]
 [[:d :e :f :g] 1.0]
 [[:f] 1.0]
 [[:x :y :z] 1.0]
 [[:w :x :y :z] 1.0]
 [[:n :o] 1.0]
 [[:f :g] 1.0]
 [[:a :b :c] 1.0]
  [[:s :t] 1.0])


(for [n (range 20)
      :let [n (rand-int (inc (count alphabet)))
            length (inc (rand-int 6))
            rand-substr (subvec alphabet
                                n
                                (min (+ n length)
                                     (count alphabet)))
            ;; swap a random symbol for something else
            ;; in the alphabet
            rand-substr (assoc rand-substr
                          (rand-int (count rand-substr))
                            (rand-nth alphabet))]
      :when rand-substr]
  [rand-substr
   (f/mean (->> (substring-confidence-2 (first outcome)
                                        rand-substr)
                (map #(nth % 2))))])

'([[:d] 1.0]
 [[:q :o] 1.0]
 [[:i] 1.0]
 [[:w :x :z :z] 1.0]
 [[:h :n :j :k :l :m] 0.8333333333333334]
 [[:f :g :l] 1.0]
 [[:m :r] 0.7]
 [[:f :o :h :i :j] 0.8]
 [[:h] 1.0]
 [[:l] 1.0]
 [[:e] 1.0]
  [[:m :n :o :p :q :j] 1.0]
 [[:i :b :c :d :e :f] 1.0]
 [[:a] 1.0]
 [[:p :w :x] 0.6666666666666666]
 [[:q :o :p] 1.0]
 [[:b :c :d :q] 0.75]
 [[:y :x] 0.5]
 [[:k :o :m :n :o :p] 0.8333333333333334]
 [[:b :r :d :e :f] 0.8800000000000001])


(let [rand-substr [:a :a :a :a]]
  [rand-substr
   (f/mean (->> (substring-confidence-2 (first outcome)
                                        rand-substr)
                (map #(nth % 2))))])
[[:a :a :a :a] 0.8125]
(let [rand-substr [:f :b :x]]
  [rand-substr
   (f/mean (->> (substring-confidence-2 (first outcome)
                                        rand-substr)
                (map #(nth % 2))))])
[[:f :b :x] 0.6666666666666666]


;; -------------------
;; use comparator v1
;;
;; - comparator v1 keeps a prediction state around (see k_fold_triangle.clj)
;; - this is not 'impressed' by the current state.
;; - So the comparotor compares the purely 'predicted' state to the world.
;; - This separates the world from the internal state enough so that we can make
;;   a judgement, whether this is a remembered sequence.
;;

(defn comparator-state
  [predictor comperator]
  {:comperator comperator
   :prediction-register (hd/->seed)
   :predictor predictor
   :t 0})

(defn update-comparator
  [{:as state
    :keys [predictor prediction-register comperator]}
   s-world]
  (-> state
      (update :t inc)
      (assoc :prediction-register (predictor s-world))
      (assoc :comperator-output
             (comperator prediction-register s-world))))

;;
;; this one has the downside that the first comparison is non sense
;; You must ignore the first comparison.
;; No way around it, if you only post-dict from 1 in the past.
;;

(defn substring-confidence-3
  [m substring-symbols]
  (let [action :right
        _ (do (recover m (hd/->empty) 0)
              (recover m (hd/->empty) 0)
              (recover m (hd/->empty) 0))]
    (->> (reductions
           update-comparator
           (comparator-state
             (fn [s]
               (recover m (hdd/clj->vsa* {action s}) 1)
               (some-> (recover m (hd/->empty) 1)
                       (hd/permute-inverse)))
             (fn [prediction s-world]
               (when prediction
                 [(hdd/cleanup* prediction) s-world
                  (hd/similarity prediction
                                 (hdd/clj->vsa*
                                  s-world))])))
           substring-symbols)
         (drop 2)
         (map :comperator-output))))

(let [rand-substr [:m :n :o :p :q :j]]
  [rand-substr
   (f/mean (->> (substring-confidence-3 (first outcome)
                                        rand-substr)
                (map #(nth % 2))))])
[[:m :n :o :p :q :j] 0.8]

(let [rand-substr [:j :k :l]]
  [rand-substr
   (f/mean (->> (substring-confidence-3 (first outcome)
                                        rand-substr)
                (map #(nth % 2))))])
[[:j :k :l] 1.0]

(defn contains-reference
  [v subv]
  (= subv
     (subvec v
             (.indexOf v (first subv))
             (min (count v)
                  (+ (.indexOf v (first subv))
                     (count subv))))))

(def contains-classification-outcome
  (doall
   (for [n (range 30)
         :let [n (rand-int (inc (count alphabet)))
               length (+ 2 (rand-int 6))
               rand-substr (subvec alphabet
                                   n
                                   (min (+ n length)
                                        (count alphabet)))]
         :when (< 1 (count rand-substr))
         :let [ ;; swap a random symbol for something
               ;; else in the alphabet
               rand-substr (if (zero? (rand-int 2))
                             rand-substr
                             (assoc rand-substr
                                    (rand-int (count
                                               rand-substr))
                                    (rand-nth alphabet)))]]
     (let [outcome (f/mean (->> (substring-confidence-3
                                 (first outcome)
                                 rand-substr)
                                (map #(nth % 2))))]
       {:cog-contains? (< 0.95 outcome)
        :contais-reference?
        (contains-reference alphabet rand-substr)
        :outcome outcome
        :rand-substr rand-substr}))))

(filter
 (fn [[a b]] (not= a b))
 (map (juxt :cog-contains? :contais-reference?) contains-classification-outcome))
'()

;;
;; It solves the substring problem
;;
;; Next: Make empirical study what the capacity is, with parameters for SDM and alphabet length etc.
;;


;; -----------------------------------------------
;; what comes after z?

(let [[m lst] outcome]
  (let [state :z
        action :right
        _ (do (recover m (hd/->empty) 0)
              (recover m (hd/->empty) 0)
              (recover m (hd/->empty) 0))
        prediction
        (hd/permute-inverse
         (do (recover m (hdd/clj->vsa* {action state}) 0)
             (recover m (hd/->empty) 1)))]
    (hdd/cleanup* prediction)))
'(:z)


;;
;; comes out as :z right now, because `update-world` defaults to 'state', if presented with
;; a state-input that is not in the world.
;;
;; - I did this because I thought it makes sense that the world stays the same, if the action is non-sense.
;;
;; Options that come to mind:
;;
;; - right of z is z, like here right now
;; - right of z is non-sense, which can be interpreted as a boundary [:z :right] -> non-sense,
;;   querying with non-sense will not go anywhere
;; - right of z is a end of sequence token, :eos
;; - right of z is a special token for :nothing, or :boundary (in this version left of a and right of z is the same)
;;


;; -----------------------------------------------
;; Predicting a sequence from a letter
;;

(let [[m lst] outcome]
  (let [state (rand-nth alphabet)
        action :right
        _ (do (recover m (hd/->empty) 0)
              (recover m (hd/->empty) 0)
              (recover m (hd/->empty) 0))]
    (map hdd/cleanup*
      (reductions
        (fn [state _]
          (hd/permute-inverse
            (recover m (hdd/clj->vsa* {action state}) 1)))
        (hdd/clj->vsa* state)
        (range 7)))))

'((:f) (:f) (:f) (:f) (:f) (:f) (:f) (:f))
'((:k) (:k) (:k) (:k) (:k) (:k) (:k) (:k))
'((:i) (:i) (:j) (:j) (:k) (:k) (:k) (:k))

;; ... that is lame

(let [[m lst] outcome]
  (let [state (rand-nth alphabet)
        action :right
        _ (do (recover m (hd/->empty) 0)
              (recover m (hd/->empty) 0)
              (recover m (hd/->empty) 0))]
    (map hdd/cleanup*
      (reductions
        (fn [state _]
          (recover m (hdd/clj->vsa* {action state}) 0)
          ;; flow time 1 more
          (hd/permute-inverse (recover m (hd/->empty) 1)))
        (hdd/clj->vsa* state)
        (range 7)))))

'((:c) (:d) (:e) (:f) (:g) (:h) (:i) (:j))
'((:p) (:q) (:r) (:s) (:t) (:u) (:v) (:w))
'((:h) (:i) (:j) (:k) (:l) (:m) (:n) (:o))


;; try with 2 'leading' / 'priming' elements

(let [[m lst] outcome]
  (let [state-xs (rand-nth
                   (into [] (partition-all 2 alphabet)))
        action :right
        _ (do (recover m (hd/->empty) 0)
              (recover m (hd/->empty) 0)
              (recover m (hd/->empty) 0))
        recover (fn [e]
                  (-> (recover-1 m e 1)
                      :result
                      ;; sdm/torch->jvm
                      ;; (dtt/->tensor :datatype :int8)
                  ))]
    (let [s (recover (hdd/clj->vsa* {action (first
                                              state-xs)}))]
      (map hdd/cleanup*
        (map hd/permute-inverse
          (map (fn [t]
                 (-> t
                     sdm/torch->jvm
                     (dtt/->tensor :datatype :int8)))
            (reductions
              (fn [state elm]
                (recover (hdd/clj->vsa* {action elm})))
              s
              (concat (rest state-xs)
                      (repeatedly 5 hd/->seed)))))))))

'((:s) (:t) (:u) (:m :y :j) (:n) (:t) (:c))
;; ... first 2 + 3rd make sense and the rest is random.
;; makes sense on it's own terms.


;; (how to retrieve sequences is at sdm.clj: ->k-fold-memory)
;;



;; ----------------------
;; Discussion:
;;
;; - can encode sequences
;; - can use the sequence model as non-deterministic finite state automaton
;;
;; Outlook:
;;
;; Current goals:
;; - Build up to (partially) solving the copycat domain (Hofstadter + Mitchel 1988)
;; - HDC interpreters:
;;   one idea is using non-deterministic finite state automata,
;;   perhaps the presence of an element in the sequence can be an instruction to the interpreter
;;   and so forth.
;; - I feel like there must be a flow control implementation by selecting one of possible trajectories,
;;   where the trajectories represent process paths.
;;   I.e. 'collapsing' the non-deterministic finite state automata into one of it's possible states,
;; - I feel that the flow control primitives of such a framework should allow for branches to be
;;   interpreted in superposition, this would be the 'computing in superposition' way.
;;
;;
;;
;;
;;
;; - find a way to encode hierachical trajectories
;; - Idea: Use a triangle world as 'analogy' to label subsequences
;;
;;
;;
;;                        +--------+---------+-------+
;;                        |    a   |    b    |   c   |        triangle world, 3 states
;;                        +--------+---------+---+---+
;;                            |         |        |
;;                            |         |        |             some kind of mapping
;;                   +--------+         v        +-----+
;;                   v                                 v
;;            a, b, c, d, e, ...      j, k, l,..  s, t, u, ...  terminal sequences of alphabet
;;
;;
;; - i.e. the 'b' of triangle domain is allowed to be analogous to the subsequence starting with 'j' in alphabet domain.
;; - the terminal sequences get order, by being mapped to the smaller triangle world.
;;
;;
;; - snippets of trajectories could perhaps stand for action sequences
;; - my vision is something that commposes trajectories and mappings
;;
;; - Roughly, a computational system that works by juggleling trajectories and mappings between trajectories,
;;   juxtoposing them and concatenating them, then computing generilized continuations, and somehow collapsing them.
;; - One cool thing is that with hdc, the elements are allowed to be sequences, trees, finite-state automatons etc themself (see data.clj).
;;
;;
;;
;; - allow 'analogy structures' to be 'action structures'
;;   [:right :right :left :right], an action seq - abstracted from the terminal elements.
;;
;;   Thus, I see this as a 'analogy structure', right now that doesn't do much, but you see that [:right :right :right] would produce
;;   abc in the 'a' domain and jkl in the 'j' domain.
;;   (Id did exactly this at what_is_the_abc_that_starts_with_j.clj)
;;
;;   You will need composite actions for this to be useful.
;;
;;
;;
;; - Combining this with higher order trajectory analogies (mappings), this starts looking like a way to make 'grammer', where the action terminals are syntax,
;;   And the higher order sequences constrain the syntax.
;; - And in this framework, action trajectories, grammers, and analogy structures would be the same thing.
;;
;;
;;
;; - how to have on-the-fly mappings? It sounds like exactly the problem you solve with neuronal ensembles glueing together temporarily
;; - similar to fast-weights [Hinton], synapsembles [Buzsáki], the excitability ensembles of Yuste
;; - the problem is reminiscent of 'working memory' or 'scratchpad'
;; - My idea is to try model an element called a 'glue space'.
;; - A glue space can be wiped (killed), created (then fresh). Can be used as autoassocitive or heteroassociate memory.
;; - Optionally, it's a sequence memory.
;; - I can implement such a glue space with a temporary SDM.
;;
;;
