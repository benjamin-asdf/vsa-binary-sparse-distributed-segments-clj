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
;; More: blog, lit
;;
;; Some variations exist with slightly different properties.
;;
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

(defn ->hv
  ([] (->hv default-opts))
  ([{:bsdc-seg/keys [segment-count segment-length N]}]
   (let [indices (repeatedly segment-count
                             #(fm.rand/irand segment-length))
         indices (f/+ indices
                      (f/* (range segment-count)
                           segment-length))
         v (dtype/alloc-zeros :int8 N)]
     (doseq [i indices] (dtype/set-value! v i 1))
     (dtt/->tensor v))))

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
;; For VSA, we need a similarity meassure of HDVs. Cosine is good for many architectures, but not
;; for sparse binary.
;; I take the overlap and normalize to 0 and 1. Seems to work.
;;
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
   (/ (f/sum (f/bit-and a b)) segment-count)))


;;
;; -------------------
;; 3. Permutation
;; -------------------
;;
;; Permutation allows us to 'hide' data in an HDV `A`, (`protect`).
;; This is useful:
;;
;; - To represent the quotation `'A` of `A`
;; - To represent causality, or direction
;;
;; E.g.
;; (bundle A (permute B))
;; For 'B follows A'
;;
;; More generally, this can be used to encode sequences with a sumset (see below), by permuting the ith element ith times.
;;
;; - To randomize vectors (random permutation) (not sure how relevant with BSDC)
;;

(defn permute-n
  "Shift the segments of `a` circularly."
  ([a n] (permute-n a n default-opts))
  ([a n {:bsdc-seg/keys [segment-length]}]
   ;; (direction doesn't actually matter of course, just that inverse is the inverse)
   (dtt/rotate a [(* -1 n segment-length)])))

(defn permute-inverse-n
  "See [[permute-n]]"
  [a n]
  (permute-n a (* -1 n)))

(defn permute
  "Default permute 1 time.
  See [[permute-n]]"
  [a]
  (permute-n a 1))

(defn permute-inverse
  "The inverse of [[permute]].
  See [[permute-n]]"
  [a]
  (permute-n a -1))


;; ------------
;; 4. Sumset (Bundle) Operation and Thinning
;; ------------
;;
;; Sumset A and B
;; Take the elementwise sum.
;; The result vector `C` = `A` + `B` is similar to both.
;; Because you can 'get out' both A and B and they are unorded this is a 'set'.
;;
;; This is also called `bundle`. As in bundling the information into a single representation.
;;
;; Thinning:
;; To keep it sparse, we usually want to thin.
;; (Although it is cool that we can sum multiple vectors without thinning.
;; And permutation and bind also work on denser vectors).
;;
;;

;; bundle is easy
(def bundle
  "Returns the elementwise sum of input the vectors.
  Output vector resembles all input vectors, unordered.
  Hence this is also called `sumset`.
  "
  f/+)

(defn thin-pth-modulo
  "Returns a new thinned vector of `a` where 1 non-zero bit per segment in `a` is left over.

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

  Trust me, you'll get a toaster out of it

  How many you can bundle before things water down is called the `capacitiy`.
  [blog]



  - preserves distance

(let [a (->hv) b (->hv) ab (thin (bundle a b)) c (->hv)]
  (= (similarity a ab)
     (similarity (bind a c) (bind ab c))))

  => true


  - left distrubutes over addition

(let [a (->hv) b (->hv) c (->hv)]
  (=
   (bind a (f/+ b c))
   (f/+ (bind a b) (bind a c))))

  => true

  This is not a self-inverse bind. To unbind, you use [[unbind]].

  Implementation:
  ---------------

   +----------+----------+----------+--------------+
   | _ _ 1 _  |          |          |              |  A
   +-----^----+----------+----------+--------------+
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

     shift
     ---->
      idx * alpha

    [ _ _ _ 1 ] ,


              2. shift the segment-j of B by idx * alpha circularly
                 alpha is 1 by default (so you move to the right)
                 alpha = -1 does the reverse shift and unbinds.


     [ shift,   shift,    shift,  ...  ]

     -> call it C

  This is a segment-wise `permutation` of B.

  See [[unbind]]
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


(let [color (->hv)
      red (->hv)
      kvp (bind color red)]
  (similarity (unbind kvp color) red))
1.0



(let [color (->hv)
      red (->hv)]
  [(similarity (unbind (bind red color) color) red)
   (similarity (unbind (bind color red) color) red)])











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

;; but it is not associative:

(let [a (->hv)
      b (->hv)
      c (->hv)]
  (similarity
   (bind (bind a b -1) c -1)
   (bind a (bind b c -1) -1)))

(let [a [0 0 1]
      b [1 0 0]
      opts
      {:bsdc-seg/segment-length 3
       :bsdc-seg/segment-count 1
       :bsdc-seg/N 3
       }]

  ;; (bind (bind a b -1 ) c -1)
  [
   (bind a b 1 opts)
   (bind a b -1 opts)])

;; (let [a (->hv)
;;       b (->hv)
;;       ab (thin (bundle a b))
;;       banana (->hv)
;;       apple (->hv)
;;       fruit (thin (bundle banana apple))
;;       mapping (bind ab fruit)]
;;   [(similarity a ab)
;;    ;; preserves distance
;;    (similarity (bind mapping a) (bind mapping ab))
;;    (similarity (unbind mapping fruit) a)
;;    (similarity (unbind mapping fruit) b)
;;    ;; meaningless
;;    (similarity (unbind mapping fruit) banana)
;;    ;; 👌
;;    (similarity (unbind mapping ab) banana)])





(let [a (->hv) b (->hv) c (->hv)]
  (=
   (bind a (f/+ b c))
   (f/+ (bind a b) (bind a c))))




(let [a (->hv) b (->hv) ab (thin (bundle a b)) c (->hv)]
  (= (similarity a ab)
     (similarity (bind a c) (bind ab c))))

true
