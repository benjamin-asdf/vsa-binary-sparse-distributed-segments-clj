(ns sparser
  (:require [bennischwerdtner.hd.binary-sparse-segmented :as
             hd]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
            [tech.v3.parallel.for :as pfor]
            [tech.v3.datatype.argops :as dtype-argops]
            [tech.v3.datatype.functional :as f]
            [clojure.math.combinatorics :as combo]
            [bennischwerdtner.sdm.sdm :as sdm]))


;;
;; https://youtu.be/NgMWHEC2A4g?si=wzQ_tmUx879Oxsuj
;;
;; Kanerva talks about the properties of a sparse vector with 20 random bits.
;; (here segmented)
;;

;; I can use 20 segments that are 500 wide  (N = 10.000)

;; (/ (long 1e4) 20)
;; 500
;; They are all 500 wide.

(alter-var-root #'hd/default-opts
                (constantly
                 (let [dimensions (long 1e4)
                       segment-count 20]
                   {:bsdc-seg/N dimensions
                    :bsdc-seg/segment-count segment-count
                    :bsdc-seg/segment-length
                    (/ dimensions segment-count)})))


(let [lut (atom {})]
  ;; "encountering a symbol" since symbol and value are
  ;; interchangeable in hdc (Kanerva 2009), why not
  ;; simply call it `prototype`
  ;;
  (defn ->prototype
    "This also stores the symbol in content addressable memeory.

  [[known]] will return the cleaned up symbol.
  "
    [sym]
    (or (@lut sym)
        (let [v (hd/->hv) _ (swap! lut assoc sym v)] v)))
  (defn cleanup-lookup-verbose
    ([query-v] (cleanup-lookup-verbose query-v 0.09))
    ([query-v threshold]
     (->> (map (fn [[k v]]
                 {:k k
                  :similarity (hd/similarity v query-v)
                  :v v})
            @lut)
          (filter (comp #(<= threshold %) :similarity))
          (sort-by :similarity (fn [a b] (compare b a))))))
  (defn cleanup-lookup-value
    [query-v]
    (some->> (cleanup-lookup-verbose query-v)
             first
             :k))
  (defn cleanup-mem [] @lut))


;; ------------------------
;; Ambiguity primitives
;; ----------------------


(defn mix
  "Mix `a`, `b`, ... `arg` hypervectors, representing the superposition.

  This thins the output, if you want to mix and keep dense just use
  [[hd/bundle]] or [[f/+]].

  alias: [[possibly]].
  "
  ([a b & args] (hd/thin (apply f/+ a b args)))
  ([a b] (hd/thin (hd/bundle a b))))

(def possibly mix)

;; Not really sure about this yet.
;; Using both vectors to find a third that is dissimilar to both `is` bind.
;; And then I permute to distinguish it from bind.
(defn neither "Quoted bind." [a b]
  (hd/permute (hd/bind a b)))

(defn roughly
  "Returns a weaker hypervector version of `a`.

  `amount-of-a`: If 1, this is identity, if 0 this returns a zero vector.

  See [[mostly]].
  "
  [a amount-of-a]
  (hd/weaken a (- 1 amount-of-a)))

(defn mostly
  "Returns a hypervector that is mostly `a` and a little bit `b`.

  `amount-of-b`:

  If 1, this is equalent to a bundle, and returns a vector
  that is equally similar to both `a` and `b` (~0.5 each).
  I.e. that is in the middle between the 2.

  If 0, this retuns `a`.

  See [[roughly]].
  "
  ([a b] (mostly a b 0.3))
  ([a b amount-of-b]
   (hd/thin (hd/bundle a (roughly b amount-of-b)))))

(defn without
  "Returns a (potentially super sparse) version of `a` with all of `b` removed.

  Super sparse means sparser than per [[hd/maximally-sparse?]].

  Alias: [[impossibly]].
  "
  [a b]
  (hd/thin (f/- a b)))

(def impossibly without)

(defn non-sense
  "Returns a fresh random hypervector.

  Aliases: [[hd/->hv]], [[hd/->seed]]
  "
  []
  (hd/->hv))

;; I think there is something deep about the concept that
;; non-sense and gensym are the same operation
;; P. Kanerva: "Randomness is the path of least assumption".
;; G. Buzsáki: "Nothing is new for the brain."
(def create non-sense)

(defn nothing
  "Return an empty hypervector.

  This is similar to no other hypervector, not even itself.
  (because similarity is overlap of non 0 bits here).

  It is still [[=]] to iteslf.

  (= (nothing) (nothing))
  true
  (hd/similarity (nothing) (nothing))
  0.0
  (hd/similarity (nothing) (create))
  0.0

  "
  []
  (hd/->empty))


;; ... 'everything' would also make sense
(defn everything
  "Returns a hypervector similar to every other one.

(let [a (non-sense)
      b (non-sense)
      c (neither a b)]
  [(hd/similarity a (everything))
   (hd/similarity b (everything))
   (hd/similarity c (everything))])
[1.0 1.0 1.0]

  Sounds like epilepsy to me.

  This would also activate every single address location of a sparse distributed memory,
  if used as address word.

  See [[hd/similarity]].
  "
  []
  (hd/->ones))






(comment
  (let [a (hd/->hv)
        b (hd/->hv)]
    ;; then it is 50:50
    [(hd/similarity a (mostly a b 1.0))
     (hd/similarity b (mostly a b 1.0))])
  [0.4 0.6]
  (let [a (hd/->hv)
        b (hd/->hv)]
    ;; this is mostly a
    [(hd/similarity a (mostly a b 0.5))
     (hd/similarity b (mostly a b 0.5))])
  [0.75 0.25]
  ;; I guess 0.3 is at the limit of still being similar
  ;; to b
  (let [a (hd/->hv)
        b (hd/->hv)]
    ;; this is mostly a
    [(hd/similarity a (mostly a b 0.3))
     (hd/similarity b (mostly a b 0.3))])
  [0.77 0.19])

(cleanup-lookup-verbose
 (hd/unbind
  (hd/thin (apply hd/bundle
                  [(hd/bind (->prototype :a) (->prototype :x))
                   (hd/bind (->prototype :b) (->prototype :y))
                   (hd/bind (->prototype :c)
                            (->prototype :x))]))
  (->prototype :c)))

;; ({:k :x, :similarity 0.35, :v #tech.v3.tensor<int8> [10000] [0 0 0 ... 0 0 0]})
(comment
  (hd/similarity (mix (->prototype :a)
                      (->prototype :b)
                      (->prototype :c)
                      (->prototype :d))
                 (->prototype :d))
  0.3)







(defn non-sense
  "Returns a fresh random hypervector.

  Aliases: [[hd/->hv]], [[hd/->seed]]
  "
  []
  (hd/->hv))

;; I think there is something deep about the concept that
;; non-sense and gensym are the same operation
;; P. Kanerva: "Randomness is the path of least assumption".
;; G. Buzsáki: "Nothing is new for the brain."
(def create non-sense)

(defn mix
  "Mix `a`, `b`, ... `arg` hypervectors, representing the superposition.

  This thins the output, if you want to mix and keep dense just use
  [[hd/bundle]] or [[f/+]].

  alias: [[possibly]].
  "
  ([a b & args] (hd/thin (apply f/+ a b args)))
  ([a b] (hd/thin (hd/bundle a b))))

(def possibly mix)

;; Not really sure about this yet.
;; Using both vectors to find a third that is dissimilar to both `is` bind.
;; And then I permute to distinguish it from bind.
(defn neither "Quoted bind." [a b]
  (hd/permute (hd/bind a b)))

(defn vanishingly
  "Returns a weaker hypervector version of `a`.

  `amount-of-a`: If 1, this is identity, if 0 this returns a zero vector.

  See [[mostly]], [[nothing]], [[roughly]].
  "
  ([a] (vanishingly 0.2))
  ([a amount-of-a]
   (hd/weaken a (- 1 amount-of-a))))

(defn mostly
  "Returns a hypervector that is mostly `a` and a little bit `b`.

  `amount-of-b`:

  If 1, this is equalent to [[mix]], and returns a vector
  that is equally similar to both `a` and `b` (~0.5 each).
  I.e. that is in the middle between the 2.

  If 0, this retuns `a`.

  If 0 < `amount-of-b` < 1, this returns a point between `a` and `b`
  that is closer to `a` than to `b`.

  See [[roughly]].
  "
  ([a b] (mostly a b 0.4))
  ([a b amount-of-b] (mix a (vanishingly b amount-of-b))))

(comment

  (for [n (range 5)]
    (let [a (create)
          amount (rand)
          c (vanishingly a amount)]
      [:amount amount :similarity (hd/similarity a c)]))

  '([:amount 0.03376090805508081 :similarity 0.05]
    [:amount 0.27461511894016855 :similarity 0.25]
    [:amount 0.27046438370634807 :similarity 0.3]
    [:amount 0.592055396689516 :similarity 0.55]
    [:amount 0.6953485647339649 :similarity 0.7]))

(defn roughly
  "Returns a hypervector similar to `a`, but with random noise mixed in.

  Think 'merely roughly `a` to a certain degree'.

  `amount-of-noise`: If 0, this is identity. If 1, this returns a vector with 0.5 similarity to `a`.

  See [[mostly]], [[vanishingly]], [[non-sense]], [[create]].
  "
  ([a] (roughly a 0.2))
  ([a amount-of-noise]
   (mostly a (non-sense) amount-of-noise)))

(comment
  (let [a (create)]
    [(hd/similarity a (roughly a 0.2))
     (hd/similarity a (roughly a 1))])
  [0.9 0.5])

(comment
  (for [n (range 5)]
  (let [a (create)
        b (create)
        c (mostly a b)]
    [(hd/similarity a c) (hd/similarity b c)]))

  '([0.85 0.15] [0.8 0.2] [0.8 0.2] [0.85 0.15] [0.85 0.15]))

;; alias 'absolutely-not'?
(def impossibly
  "Returns a hypervector representing the 'removal' of the arguments.

  Note that this returns a hypervector with negative bits.
  This will be 'similar' to it's negation.
  (because we count the non zero bit overlap).

  This is different from representing the absence of the arguments.

  [[nothing]] is the commplete absence.
  [[without]] is the absence of a part in a whole.

  [[hd/similarity]].
  "
  (comp f/- f/+))

(comment
  (let [a (create)
        b (create)
        c (hd/bind a b)
        c2 (hd/bind a (impossibly b))]
    [(f/sum c2)
     ;; strangely 'similar'
     (hd/similarity b (hd/unbind c2 a))
     (= (impossibly b) (hd/unbind c2 a))
     (hd/similarity (mix a (impossibly b)) a)
     (hd/similarity (mix a (impossibly b)) b)])
  [-20.0 1.0 true 1.0 0.0])

(defn without
  "Returns a (potentially super sparse) version of `a` with all of `b` removed.

  Super sparse means sparser than per [[hd/maximally-sparse?]].

  See [[impossibly]].
  "
  [a b]
  (mix a (impossibly b)))

(defn nothing
  "Returns the empty hypervector.

  Nothing is similar to no other hypervector, not even itself.
  (because similarity is overlap of non 0 bits here).

  It is still [[=]] to iteslf.

  (= (nothing) (nothing))
  true

  (hd/similarity (nothing) (nothing))
  0.0

  (hd/similarity (nothing) (create))
  0.0

  See [[hd/similarity]], [[non-sense]].
  "
  []
  (hd/->empty))

(comment
  (= (nothing) (nothing))
  true

  (hd/similarity (nothing) (nothing))
  0.0

  (hd/similarity (nothing) (create))
  0.0)

;; ... 'everything' would also make sense
;; If used as a permutation wiring, perhaps this should be equal to a fully connected neural net layer
(defn everything
  "Returns a maximally dense hypervector that is similar to all other hypervectors.

  This would also activate every single address location of a sparse distributed memory,
  if used as address word.

  (let [a (non-sense)
        b (non-sense)
        c (neither a b)]
    [(hd/similarity a (everything))
     (hd/similarity b (everything))
     (hd/similarity c (everything))])
  [1.0 1.0 1.0]


  See [[hd/similarity]].
  "
  []
  (hd/->ones))




(comment

  (let [a (create)
        b (create)
        c (mix a b)]
    [(hd/similarity a b)
     (hd/similarity a c)
     (hd/similarity a (mix a (impossibly a)))])


  ;; ??
  ;;
  ;; certainly
  ;; absolutely
  ;; sharply
  ;; never
  ;;
  ;; ??
  ;; 'neccessary conncetion'
  ;;



  (let [a (non-sense)
        b (non-sense)
        c (neither a b)]
    [(hd/similarity a (everything))
     (hd/similarity b (everything))
     (hd/similarity c (everything))])
  [1.0 1.0 1.0]

  )
