(ns ambiguity
  (:require
   [bennischwerdtner.hd.binary-sparse-segmented :as hd]
   [tech.v3.tensor :as dtt]
   [tech.v3.datatype.functional :as f]))


;; ------------------------
;; Ambiguity primitives *** WIP ***
;; ----------------------
;;
;; The Cortex is an information mixing machine - V. Braitenberg
;;
;;
;; Update:
;; --------
;; This preceeds src/bennischwerdtner/hd/data.clj
;;
;; - 'without' can be replaced with set difference
;; - mix only works well when you work with small amont of seed vectors. (thinnig quickly loses signal strenght)
;; - you probably want to use the set functions generally instead
;; - vanishingly, roughly, mostly, impossibly, nothing, everything, non-sense make sense conceptually
;;   but perhaps I can find an alternative to 'thin' the whole thing.
;; - something more like alternative set functions, expanding the vacubulary 'union', 'difference' to include mix-states
;; - would be interesting to program in superposition, utilzing ambiguity
;;


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
  ([a] (vanishingly a 0.5))
  ([a amount-of-a]
   (hd/weaken a (- 1 amount-of-a))))

(comment
  (let [a (create)] (hd/similarity a (vanishingly a)))
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
  [0.9 0.5]

  )

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
  This is different from representing the absence of the arguments.

  [[nothing]] is the commplete absence.
  [[without]] is the absence of a part in a whole.

  [[hd/similarity]].
  "
  (comp dtt/->tensor f/- f/+))

(comment
  (let [a (create)
        b (create)
        c (hd/bind a b)
        c2 (hd/bind a (impossibly b))]
    [(f/sum c2)
     (hd/similarity b (hd/unbind c2 a))
     (= (impossibly b) (hd/unbind c2 a))
     (hd/similarity (mix a (impossibly b)) a)
     (hd/similarity (mix a (impossibly b)) b)])
  [-100.0 0.0 true 0.99 0.0])

(defn without
  "Returns a (potentially super sparse) version of `a` with all of `b` removed.

  Super sparse means sparser than per [[hd/maximally-sparse?]].

  See [[impossibly]].
  "
  [a b]
  (mix a (impossibly b)))

(comment
  (let [a (create)
        b (create)
        c (mix a b)]
    [(hd/similarity a c)
     (hd/similarity b c)
     (hd/similarity a (without c b))
     (hd/similarity b (without c b))])
  [0.4 0.6 0.4 0.0])

(defn nothing
  "Returns the empty hypervector.

  'Nothing' is similar to no other hypervector, not even itself.
  (because similarity is overlap of non 0 bits here).

  It is still [[=]] to iteslf.

  (= (nothing) (nothing))
  true

  (hd/similarity (nothing) (nothing))
  0.0

  (hd/similarity (nothing) (create))
  0.0

  See [[hd/similarity]], [[non-sense]], [[everything]]
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
  "Returns a maximally dense hypervector that is similar all hypervectors.
  Except if they are super sparse, like 'nothing'.

  This would also activate every single address location of a sparse distributed memory,
  if used as address word.

  (let [a (non-sense)
        b (non-sense)
        c (neither a b)]
    [(hd/similarity (neither a b) (everything))
     (hd/similarity b (everything))
     (hd/similarity c (everything))
     ;; not even everything is similar to nothing
     (hd/similarity (nothing) (everything))])
[1.0 1.0 1.0 0.0]


  See [[hd/similarity]], [[nothing]], [[non-sense]].
  "
  []
  (hd/->ones))

(comment
  (let [a (non-sense)
        b (non-sense)
        c (neither a b)]
    [(hd/similarity (neither a b) (everything))
     (hd/similarity b (everything))
     (hd/similarity c (everything))
     ;; not even everything is similar to nothing
     (hd/similarity (nothing) (everything))])
  [1.0 1.0 1.0 0.0])
