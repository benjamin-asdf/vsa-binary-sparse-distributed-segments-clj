(ns bennischwerdtner.hd.binary-sparse-segmented
  (:require
   [clojure.test :as t]
   [tech.v3.datatype.functional :as f]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [tech.v3.datatype.bitmap :as bitmap]
   [fastmath.random :as fm.rand]
   [fastmath.core :as fm]
   [tech.v3.datatype.unary-pred :as unary-pred]
   [tech.v3.datatype.argops :as dtype-argops]))

;;
;; High-Dimensional Computing with Sparse Vectors
;;
;; https://redwood.berkeley.edu/wp-content/uploads/2020/10/laiho_2015_high.pdf
;; Laiho et. al. 2015
;;
;;

;; -------------------
;; 1. Elements - The Hypervector
;; -------------------

;; The elements of high dimensional computing (HDC, also called Vector Symbolic Architectures VSA)
;; are the hypervectors (HDV).
;; These are vectors with large number dimensions (100++).
;;

;;
;; Binary Segmented Sparse
;;

;;  HDV:
;;
;;    <------ N bits wide, typically 10.000         --->
;;
;;  +-----------+-----------+------------+------------------+
;;  | _ _ _ _ 1 | _ 1 _ ..  |     1      |                  |
;;  +-----------+-----------+------------+------------------+
;;    s          s            s             ... segment-count (~100)
;;  <----------->
;;  segment-length (~100)
;;
;;  Each segment has 1 (random) bit non-zero.
;;
;;  In order to make the V sparse, we only set 1 bit per segment.
;;  We also implement a `thin` operation (see below) for when we have vectors of higher density.
;;

(def default-opts
  "Default opts for (binary sparse distributed, segmented) BSDC-SEG vector operations."
  (let [dimensions (long 1e4)
        ;; Rachkovskij (2001) showed that this value
        ;; works well, therefore we use it
        density-probability (/ 1 (fm/sqrt dimensions))
        ;; segment count == non-zero bits count
        segment-count (long (* dimensions
                               density-probability))]
    {:bsdc-seg/N dimensions
     :bsdc-seg/density-probability density-probability
     :bsdc-seg/segment-count segment-count
     :bsdc-seg/segment-length (/ dimensions
                                 segment-count)}))


(defn indices->hv
  "Returns a segmented hypervector with `indices`
  set to 1, segmentwise.

  "
  ([indices] (indices->hv indices default-opts))
  ([indices
    {:bsdc-seg/keys [segment-count segment-length N]}]
   (let [indices (f/+ indices
                      (f/* (range segment-count)
                           segment-length))
         v (dtype/alloc-zeros :int8 N)]
     (doseq [i indices] (dtype/set-value! v i 1))
     (dtt/->tensor v))))

(defn hv->indices
  "Returns the segment representation of `hv`.

  This is a list of indices for each segement.

  example hv with N = 6, segment-length = 3


  hv:            segment representation:
  [ 0 1 0 , 1 0 0 ]  ( 1, 0 )

  Assumes `a` is a maximally sparse vector.
  "
  ([a] (hv->indices a default-opts))
  ([a {:bsdc-seg/keys [segment-count segment-length N]}]
   (-> (dtt/reshape a [segment-count segment-length])
       (dtt/reduce-axis (fn [segment] (dtype-argops/argmax segment))))))

(comment
  (hv->indices [0 1 0 1 0 0] {:bsdc-seg/N 6 :bsdc-seg/segment-count 2 :bsdc-seg/segment-length 3})
  [1 0]
  (hv->indices (->hv)))


(defn ->hv
  "
  Returns a fresh, random hypervector - the element of VSA.

  __Binary Segmented Sparse__ :


    <------ N bits wide, typically 10.000         --->

  +-----------+-----------+------------+------------------+
  | _ _ _ _ 1 | _ 1 _ ..  |     1      |                  |
  +-----------+-----------+------------+------------------+
    s          s            s             ... segment-count (~100)
  <----------->
  segment-length (~100)


  Each segment has 1 (random) bit non-zero.

  In order to make it sparse, we only set 1 bit per segment.

  For operation, see:
  [[thin]], [[maximally-sparse?]], [[similarity]], [[bundle]], [[bind]], [[unbind]]
  "
  ([] (->hv default-opts))
  ([{:bsdc-seg/keys [segment-count segment-length N]}]
   (indices->hv (repeatedly segment-count
                            #(fm.rand/irand
                              segment-length)))))

(def ->seed ->hv)

;; good enough for my needs
(defn hv?
  "Returns true when `x` is a hypervector."
  ([x] (hv? x default-opts))
  ([x {:bsdc-seg/keys [N]}]
   (and (dtt/tensor? x) (= N (dtype/ecount x)))))



(defn maximally-sparse?
  "Returns true if `a` is maximally sparse.
  Then each segment normally has 1 non-zero bit."
  ([a] (maximally-sparse? a default-opts))
  ([a {:bsdc-seg/keys [segment-count]}]
   (= segment-count (long (f/reduce-+ (f/not-eq 0 a))))))


;;
;; -------------------
;; 2. Similarity
;; -------------------
;;
;; For VSA, we need a similarity measure of HDVs. Cosine is good for many architectures, but not
;; for sparse binary.
;; I take the overlap and normalize between 0 and 1. Seems to work.
;;
;;

(defn similarity
  "
  Returns a `similarity` of `a` and `b`.
  I take the overlap and normalize it between 0 and 1.

  0 is not similar.
  1 is similar.

  Can also go above 1, if vectors are denser than maximal sparse.

  http://www.arxiv.org/abs/2001.11797
  "
  ([a b] (similarity a b default-opts))
  ([a b {:bsdc-seg/keys [segment-count]}]
   (/
    ;; (f/sum (f/bit-and a b))
    (f/sum
     (f/bit-and
      (f/not-eq 0 a)
      (f/not-eq 0 b)))
    segment-count)))


;;
;; -------------------
;; 3. Permutation
;; -------------------
;;

(defn permute-n
  "Returns a new vector where each segment of `a` is circularly by `n`."
  ([a n] (permute-n a n default-opts))
  ([a n {:bsdc-seg/keys [segment-length segment-count N]}]
   (-> a
       (dtt/reshape [segment-count segment-length])
       (dtt/map-axis
        (fn [segment] (dtt/rotate segment [n])))
       (dtt/reshape [N]))))

(defn permute
  "
  Returns the canonical permutation of `a`.

  Permutation allows us to 'hide' data in an HDV `A`, (also called `protect`).
  This is useful:

  - To represent the quotation `'A` of `A`
  - To represent causality, direction or something in a set:

  E.g.
  (bundle A (permute B))
  For 'B follows A'

  More generally, this can be used to encode sequences with a sumset (see below), by permuting the ith element ith times.

  - To randomize vectors (random permutation) (not sure how relevant with BSDC)

"
  [a]
  (permute-n a 1))

(defn permute-inverse
  "The inverse of [[permute]].
  See [[permute-n]]"
  [a]
  (permute-n a -1))

#_(t/deftest permute-inverse-test
  (doseq
      [n (range 500)]
      (let [a (->hv)]
        (t/is (= a (permute-inverse (permute a)))))))



;; ------------
;; 4. Sumset (Bundle) Operation and Thinning
;; ------------
;;
;; Sumset A and B:
;; The elementwise sum of A and B.
;; The result vector `C` = `A` + `B` is *similar to both*.
;; Because you can 'get out' both A and B and they are unorded this is a 'set'.
;;
;; This is also called `bundle`. As in bundling the information into a single representation.
;;
;; Thinning:
;; ------------
;; To keep it sparse, we usually want to thin after bundling.
;; (Although it is cool that we can sum multiple vectors without thinning.
;; - permutation and bind also work with denser vectors
;;
;;

(def bundle
  "Returns the elementwise sum of input the vectors.
  Output vector resembles all input vectors, unordered.
  Hence this is also called `sumset`.
  "
  f/+)

(defn thin-pth-modulo
  "Returns a new thinned vector derived from `a` where 1 non-zero bit per segment in `a` is left over.

  Instead of picking a random non-zero bit from the available ones per segment (which is valid),
  here, the indices themselves provide the decision, making this a deterministic operation.

  -----
  Thin:
  ----


            +---- more than 1 non-zero bit per segment (dense)
            |
            |          chose highest value indices
  +---------v-----+------|-------+---------------+---------+
  |               |      👇👇    |               |         | HDV
  | 1    1   1    |   1  2  2    |            1  | ....    |
  +-^-------------+--------------+---------------+---------+
    |    segment-1,     segment-2, ...
    |
    |  for each segment,
    |    1. find the indices of the highest value.
    |
  [ idx0 , idx1, idx2, ]  -> `segment-max-indices`
   |------+-----------|
          |
          |  2.
          v  (modulo (sum idxs) (count idxs))

     -> `chosen-index` for segment-1


  1.
  Candidate indices: For each segment, look at the indices with the highest value.
  Usually they are all 1's but after bundling there might be 2's etc.
  If there is a single 2, it would win out, else:

  2.
  Sum the indices, then take the modulo of the sum, resulting in p.
  The pth index is the chosen non-zero bit of each segment.
  Hence the name pth-modulo.

  This is an example of 'Context-Dependent Thinning' (Rachkovskij 2001)."
  ([a] (thin-pth-modulo a))
  ([a {:bsdc-seg/keys [segment-count segment-length N]}]
   (let [indices
         (->
          (dtt/reshape a [segment-count segment-length])
          (dtt/reduce-axis
           (fn [segment]
             (let [segment-max-indices
                   (dtype-argops/argfilter
                    (partial =
                             (f/reduce-max segment))
                    segment)
                   ;; 'p'
                   chosen-index
                   (segment-max-indices
                    (fm/mod
                     (long (f/sum
                            segment-max-indices))
                     (count segment-max-indices)))]
               chosen-index)))
          (f/+ (f/* (range segment-count) segment-length)))
         v (dtype/alloc-zeros :int8 N)]
     (doseq [i indices] (dtype/set-value! v i 1))
     (dtt/->tensor v))))

(defn thin
  "Returns a new thinned vector of `a` where 1 non-zero bit per segment in `a` is left over.

  Uses a 'Context-Dependent Thinning', so this is deterministic."
  ([a] (thin a default-opts))
  ([a opts] (thin-pth-modulo a opts)))


;; ------------
;; 5. Binding Operation
;; ------------
;;
;;
;; lit 1: https://www.researchgate.net/publication/299535938_High-Dimensional_Computing_with_Sparse_Vectors
;; lit 2: http://www.arxiv.org/abs/2001.11797
;;
;; (lit.org in this repository).
(defn bind
  "
  Returns a new vector `c` that represents the binding of `a` and `b`.
  C is dissimilar to both A and B.


  Make a kvp:

  (let [color (->hv)
      red (->hv)
      kvp (bind color red)]
  (similarity (unbind kvp color) red))

  => 1.0


  Make a record by bundling kvps:

  (let
     [color (->hv)
      red (->hv)
      kind (->hv)
      toaster (->hv)
      record (thin (bundle (bind color red)
                           (bind kind toaster)))

      result
      (unbind record kind)]
  (map #(similarity result %) [color red kind toaster]))

  (0.01 0.01 0.01 0.54)

  Trust me, you'll get the toaster out of it.

  How many you can bundle before things water down can be labeled `capacitiy` of the bind.
  You don't need to worry if the kvp are less than 20.

  [blog]


  This preserves distance:

(let [a (->hv) b (->hv) ab (thin (bundle a b)) c (->hv)]
  (= (similarity a ab)
     (similarity (bind a c) (bind ab c))))

  => true


  This left distrubutes over addition:

(let [a (->hv) b (->hv) c (->hv)]
  (=
   (bind a (f/+ b c))
   (f/+ (bind a b) (bind a c))))

  => true

  This is *not* a self-inverse bind. To unbind, you use [[unbind]].

  Implementation: A Segment wise shift
  ---------------

   +----------+----------+----------+--------------+
   | _ _ 1 _  | 1        |  1       |              |  A
   +-----+----+----------+----------+--------------+
         |
         |
        idx   1. obtain idx from each segment-j in A
         |
         |       idx-2     idx-3,    ...
       +-+       --+        |
       |           |        |
       |           |        |
       v           v        v

   +----------+----------+----------+--------------+
   | 1 _ _ _  |          |          |              |  B
   +----------+----------+----------+--------------+
     segment-1, ... segment-j

    |     ^
    |     |
    |     |
    +-----+
     shift by idx * `alpha`

          |
          |
          v

    [ _ _ _ 1 ] ,


              2. shift the segment-j of B by idx * alpha circularly
                 alpha is 1 by default (so you move to the right)
                 alpha = -1 does the reverse shift, it unbinds.


     [ shifted-1,   shifted-2,  shifted-3,    ...  ]    C


  This is a segment-wise `permutation` of B.

  See [[unbind]].
  "
  ([a b] (bind a b 1))
  ([a b alpha] (bind a b alpha default-opts))
  ([a b alpha
    {:bsdc-seg/keys [N segment-count segment-length]}]
   (let [smallest-index-of-max-per-segment
         (-> a
             (dtt/reshape [segment-count segment-length])
             (dtt/reduce-axis (fn [segment]
                                ;; the first max value
                                ;; index of the
                                ;; segment (as)
                                (dtype-argops/index-of
                                 segment
                                 (f/reduce-max
                                  segment)))))
         segments-shift
         (f/* alpha smallest-index-of-max-per-segment)
         segments-shift (volatile! segments-shift)
         next-shift! (fn []
                       (let [shift (first @segments-shift)
                             _ (vswap! segments-shift rest)]
                         shift))
         map-fn #(dtt/rotate % [(next-shift!)])]
     ;;
     ;; I have a list of shift numbers for each
     ;; segment now "the bits in each segment of
     ;; vector b are circularly shifted as
     ;; locations"
     ;;
     (-> (dtt/reshape b [segment-count segment-length])
         (dtt/map-axis map-fn)
         (dtt/reshape [N])))))

(defn unbind
  "
  Returns the result of unbinding `b` from `a`.

  `a` can be a 'mapping' and `b` a 'query'.

  Example:

(let [color (->hv)
      red (->hv)
      kvp (bind color red)]
  (similarity (unbind kvp color) red))
1.0

  See [[bind]].
  "
  ;; I swap this here so the mapping is left
  [a b] (bind b a -1))

(defn unit-vector
  "
  Returns the unit vector.

  The unit vector is identity element for `bind`:

  (let [a (->hv)]
    (= a (bind (unit-vector) a)))
  true

  "
  ([] (unit-vector default-opts))
  ([{:bsdc-seg/keys [segment-count] :as opts}]
   (indices->hv (repeatedly segment-count (constantly 0)) opts)))

(defn inverse
  "Returns the inverse of `a`.

  This is cool, because it is an alternative way to `unbind`.
  Thus, a query for `a` can be represented as

  `(inverse a)`

  and the consumer can use `bind` instead of `unbind`.

  (let [a (->hv)
            b (->hv)
            a-inv (inverse a)
            c (bind a b)]
        (assert (= b (bind c a-inv))))

  Binding yourself with your inverse results in the unit vector:

  (let [a (->hv)]
   (=
     (unit-vector)
      (bind a (inverse a))))
  => true


  Inverse is the inverse of itself:

  (let [a (->hv)]
    (= a (inverse (inverse a))))
  => true


  I didn't check how that works for non maximally sparse vectors.
  "
  ([a] (inverse a default-opts))
  ([a
    {:as opts
     :bsdc-seg/keys [segment-count segment-length N]}]
   (let [indices-a (-> a
                       (dtt/reshape [segment-count
                                     segment-length])
                       (dtt/reduce-axis
                        (fn [segment]
                          (dtype-argops/index-of
                           segment
                           (f/reduce-max segment)))))]
     (let [indices-c (map (fn [a]
                            (mod (- segment-length a)
                                 segment-length))
                          indices-a)]
       indices-c
       (indices->hv indices-c opts)))))



(comment
  (let [a (->hv)] (= (unit-vector) (bind a (inverse a))))
  true
  (doseq [n (range 100)]
    (let [a (->hv)
          b (->hv)
          a-inv (inverse a)
          c (bind a b)]
      (assert (= b (bind c a-inv)))))
  nil)


(comment
  ;; Bind/Unbind commutative/associative?


  ;; Don't entirely get the paper:

  ;; If both vectors are maximally
  ;; sparse (number of ones in both vectors is S), the binding
  ;; operation commutes, i.e., A ⊗ B = B ⊗ A. Interestingly, if
  ;; we set α to −1, the arithmetic properties of the binding and
  ;; unbinding operation are interchanged: binding with α = −1
  ;; does not commute (subtraction of indices does not commute),
  ;; whereas unbinding with α = −1 associates


  ;; this is commutative, correct:

  (let [a (->hv) b (->hv)]
    (similarity
     (bind a b)
     (bind b a)))

  1.0

  ;; ... and also associative?
  ;;
  ;; -------------------
  ;; -> (makes sense for sparse vectors)
  ;;
  ;; Consider:
  ;;
  ;; https://paperswithcode.com/paper/cognitive-modeling-and-learning-with-sparse#code
  ;; Zhonghao Yang 2023
  ;; (his bind implementation assumes maximally sparse vectors)
  ;;
  ;; Then:
  ;;
  (let [segment-index-c (fn [segment-index] (mod
                                             (+
                                              (segment-idx a segment-index)
                                              (segment-idx b segment-index))
                                             segment-length))])

  ;; for each segment you sum the two input vectors a and b indices, mod segment-length
  ;;
  ;; This is obviously commutative and associative
  ;; -------------------




  (let [a (->hv)
        b (->hv)
        c (->hv)
        d (->hv)]
    (similarity
     (bind (bind (bind a b) c) d)
     (bind (bind a (bind b c)) d)))

  1.0

  ;; but for unbind neither are true.
  ;; Some misunderstanding somewhere

  ;; substraction doesn't commute, correct I guess:

  (let [a (->hv)
        b (->hv)]
    (similarity
     (bind a b -1)
     (bind b a -1)))
  0.01

  ;; but it is not associative: 👈

  (let [a (->hv)
        b (->hv)
        c (->hv)]
    (similarity
     (bind (bind a b -1) c -1)
     (bind a (bind b c -1) -1))))




(comment
  ;; https://paperswithcode.com/paper/cognitive-modeling-and-learning-with-sparse#code
  ;; Zhonghao Yang 2023

  (defn bind2
    [a b {:bsdc-seg/keys [segment-count segment-length N]}]
    (let [indices-a (-> a
                        (dtt/reshape [segment-count
                                      segment-length])
                        (dtt/reduce-axis
                         (fn [segment]
                           (dtype-argops/index-of
                            segment
                            (f/reduce-max segment)))))
          indices-b (-> b
                        (dtt/reshape [segment-count
                                      segment-length])
                        (dtt/reduce-axis
                         (fn [segment]
                           (dtype-argops/index-of
                            segment
                            (f/reduce-max segment)))))]
      [indices-a indices-b]
      (let [indices-c (map (fn [a b]
                             (mod (+ a b) segment-length))
                           indices-a
                           indices-b)]
        (indices->hv indices-c
                     {:bsdc-seg/N N
                      :bsdc-seg/segment-count segment-count
                      :bsdc-seg/segment-length
                      segment-length}))))


  ;; our binds are the same for maximally sparse vectors

  (doseq [n (range 100)]
    (let [a (->hv)
          b (->hv)
          c1 (bind a b)
          c2 (bind2 a b default-opts)]
      (assert (= c1 c2))))
  nil


  (def a
    (indices->hv
     [0 0 0 0]
     {:bsdc-seg/N 16
      :bsdc-seg/segment-count 4
      :bsdc-seg/segment-length 4}))

  (def b
    (indices->hv
     [1 1 1 1]
     {:bsdc-seg/N 16
      :bsdc-seg/segment-count 4
      :bsdc-seg/segment-length 4}))

  (def b-inv (indices->hv [3 3 3 3] {:bsdc-seg/N 16 :bsdc-seg/segment-count 4 :bsdc-seg/segment-length 4}))

  (let
      [a (->hv) b (->hv) c (bind2 a b default-opts)]
      (= c (bind a b)))

  (=
   (unit-vector {:bsdc-seg/N 16 :bsdc-seg/segment-count 4 :bsdc-seg/segment-length 4})
   (bind2 b b-inv {:bsdc-seg/N 16 :bsdc-seg/segment-count 4 :bsdc-seg/segment-length 4}))

  (let [a (->hv)
        b (->hv)
        a-inv (inverse a default-opts)
        c (bind2 a b default-opts)]
    [(= (unit-vector default-opts)
        (bind2 a a-inv default-opts)) (= b (unbind c a))
     (= b (bind2 c a-inv default-opts))])
  [true true true])



;; I forget how this is called in the literature,
;; but we can drop bits from the vector.
;;
;; Similar to bundleling with random noise,
;; but making it thinner.
;;
;; 'thin' was taken, 'drop' is a clojure.core function
;;
;; 'weaken'?

(defn weaken
  "Returns a weakened vector by dropping segment bits from `a`.

  This allows one to express the notion of a mix of hypervectors,
  where they contribute to different amount.

  Weakening the effect of `b` in a bundle:

  (let [a (->hv)
        b (->hv)]
    [(similarity a (thin (bundle a (weaken b 0.5))))
     (similarity a (thin (bundle a b)))])
  [0.76 0.58]


  Mathematically, this is like drawing a line from a to b in hyperspace.
  Bundle finds a point exactly between a and b. distance(c,b) ≅ 0.5 ≅ distance(c,a).

  With the weaken operation, you can say how far to move from a to b.
  Like moving on the line between the points.

  This is deterministic.
  "
  ([a drop-ratio] (weaken a drop-ratio default-opts))
  ([a drop-ratio
    {:bsdc-seg/keys [segment-count segment-length N]}]
   ;; the indices decide how to drop,
   ;; 'context dependent weakening'
   ;;
   ;; I think I can just make a cutoff at the segment
   ;; indices. This might have some undesirable
   ;; properties and you might be better off with using
   ;; a random drop
   ;;
   ;; When comparing 2 random vectors, this obviously
   ;; has the desired effect.
   ;;
   (let [segmentwise-cutoff (fm/floor (* drop-ratio
                                         segment-length))
         indices (hv->indices a)
         v (dtype/alloc-zeros :int8 N)]
     (doall (map (fn [idx-in-seg i]
                   (when (<= segmentwise-cutoff idx-in-seg)
                     (dtype/set-value! v i 1)))
              indices
              (f/+ (f/* (range segment-count)
                        segment-length)
                   indices)))
     (dtt/->tensor v))))



(comment
  ;; factor of 0 doesn't change anything
  (let [a (->hv)]
    (= a (weaken a 0)))


  ;; "1" returns a zero vector
  (let [a (->hv)] (zero? (f/reduce-+ (weaken a 1))))

  (let [a (->hv)]
    (f/reduce-+ (weaken a 0)))
  100

  (for [n (range 10)]
    (let [a (->hv)]
      (f/reduce-+ (weaken a 0.5))))
  '(47 49 47 44 48 50 51 52 48 46)

  ;; it is deterministic
  (let [a (->hv)]
    (=
     (weaken a 0.5)
     (weaken a 0.5)))

  ;; here is the fun part...
  ;; I make a mix not of 1:1 a:b, but of 2:1 a:b
  (for
      [n (range 10)]
      (let [a (->hv)
            b (->hv)]
        [(similarity a (thin (bundle a (weaken b 0.5))))
         (similarity a (thin (bundle a b)))]))

  '([0.68 0.48]
    [0.76 0.47]
    [0.73 0.52]
    [0.78 0.58]
    [0.8 0.58]
    [0.78 0.5]
    [0.69 0.48]
    [0.72 0.5]
    [0.78 0.54]
    [0.79 0.57])


  (let [a (->hv)
        b (->hv)]
    [(similarity a (thin (bundle a (weaken b 0.5))))
     (similarity a (thin (bundle a b)))])
  [0.76 0.58])



;;
;; Bundle with ratio
;;
;; "more of a than b"
;;
