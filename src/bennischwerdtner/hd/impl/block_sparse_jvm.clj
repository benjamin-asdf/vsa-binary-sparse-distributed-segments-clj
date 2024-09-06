(ns bennischwerdtner.hd.impl.block-sparse-jvm
  (:refer-clojure :exclude [drop])
  (:require
   [tech.v3.datatype.unary-pred :as unary-pred]
   [tech.v3.datatype.functional :as f]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [fastmath.random :as fm.rand]
   [fastmath.core :as fm]
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
  #_(let [dimensions (long 1e4)
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
                                   segment-count)})
  ;; I started enjoying the daringness of making this
  ;; 20
  (let [dimensions (long 1e4)
        segment-count 20]
    {:bsdc-seg/N dimensions
     :bsdc-seg/segment-count segment-count
     :bsdc-seg/segment-length (/ dimensions
                                 segment-count)}))


(defn ->empty
  "Returns a 'zero vector' of hypervector lenght."
  ([] (->empty default-opts))
  ([{:bsdc-seg/keys [N]}]
   (dtt/->tensor (dtype/alloc-zeros :int8 N))))

(defn ->ones
  "Returns a 'ones vector' of hypervector lenght."
  ([] (->ones default-opts))
  ([{:bsdc-seg/keys [N]}]
   (dtt/->tensor (dtt/compute-tensor [N] (constantly 1) :int8))))

(defn ->rand-mask
  "Returns a 'ones vector' of hypervector lenght."
  ([chance] (->rand-mask chance default-opts))
  ([chance {:bsdc-seg/keys [N]}]
   (dtt/->tensor
    (dtt/compute-tensor [N] (fn [_] (fm.rand/flip chance)) :boolean))))


(defn indices->hv
  "Returns a segmented hypervector with `indices` set to 1, segmentwise.

  See [[hv->indices]].
  "
  ([indices] (indices->hv indices default-opts))
  ([indices
    {:bsdc-seg/keys [segment-count segment-length N]
     :keys
     [tensor-opts]}]
   (let [indices (f/+ indices (f/* (range segment-count) segment-length))
         v (dtype/alloc-zeros :int8 N)]
     (doseq [i indices] (dtype/set-value! v i 1))
     (dtt/->tensor v tensor-opts))))

(defn indices->hv*
  "Like [[indices->hv]] but [[indices]] should be a sequence of sequences.
  Each representing the non-zero indices of a segment (which are allowed to be empty)."
  ([indices] (indices->hv* indices default-opts))
  ([indices
    {:bsdc-seg/keys [segment-count segment-length N]
     :keys [tensor-opts]}]
   (let [v (dtype/alloc-zeros :int8 N)]
     ;; doing this using jvm is permissable for my needs, because
     ;; indices are so few
     (doall (map (fn [idxes seg]
                   (doseq [i idxes]
                     (dtype/set-value! v (+ i seg) 1)))
              indices
              (f/* (range segment-count) segment-length)))
     (dtt/->tensor v tensor-opts))))

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

(defn hv->indices*
  "Returns a segment representation of `hv`, which is allowed
  to have multiple indices per segment.

  This returns a sequence of indices vectors for each segment.

  The difference to [[hv->indices]] is that it doesn't take the argmax
  for each segment, it returns all segment indices with more than 0 value.

  See [[hv->indices]].
  "
  ([a] (hv->indices* a default-opts))
  ([a {:bsdc-seg/keys [segment-count segment-length N]}]
   (-> (f/< 0 a)
       (dtt/reshape [segment-count segment-length])
       (dtt/reduce-axis (fn [segment]
                          (unary-pred/bool-reader->indexes
                           segment))
                        -1
                        :vector))))


;; BSC Fully Distributed Representation Kanerva 1997
;; Binding and Normalization of Binary Sparse Distributed Representations by Context-Dependent Thinning
;; Rachkovskij, Kussul
;; -> it is interesting that they have a bind version that preserves similarity of the bound vectors.
;; This would be another thing I want to try out next.
;;


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

  This is also called a Sparse Block-Code.

  For operation, see:
  [[thin]], [[maximally-sparse?]], [[similarity]], [[bundle]], [[bind]], [[unbind]]
  "
  ([] (->hv default-opts))
  ([{:bsdc-seg/keys [segment-count segment-length N] :as opts}]
   (indices->hv (repeatedly segment-count
                            #(fm.rand/irand
                              segment-length))
                opts)))

;; This also called a seed vector in the literature
(def ->seed ->hv)

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
  I take the overlap where they are above 0 and normalize it between 0 and 1.

  0 is not similar.
  1 is similar.

  Can also go above 1, if vectors are denser than maximal sparse.

  http://www.arxiv.org/abs/2001.11797
  "
  ([a b] (similarity a b default-opts))
  ([a b {:bsdc-seg/keys [segment-count]}]
   (/ (f/sum (f/bit-and (f/< 0 a) (f/< 0 b)))
      segment-count)))


;;
;; -------------------
;; 3. Permutation
;; -------------------
;;

(defn permute-n
  "Returns a new hdv where `a` is blockwise permuted by `n` segments."
  ([a n] (permute-n a n default-opts))
  ([a n {:bsdc-seg/keys [segment-length segment-count N]}]
   (dtt/rotate a [(* n segment-length)])))

(defn permute-block-local-n
  "Returns the block local circular convolution of `a` by `n`.
  This is the same as binding with a known offset.

  See [[unit-vector-n]].
  "
  ([a n] (permute-block-local-n a n default-opts))
  ([a n {:bsdc-seg/keys [segment-length segment-count N]}]
   (-> a
       (dtt/reshape [segment-count segment-length])
       (dtt/map-axis (fn [segment]
                       (dtt/rotate segment [n])))
       (dtt/reshape [N]))))

(defn permute
  "
  Returns the canonical permutation of `a`.

  Permutation allows us to 'hide' data in an HDV `A`, (also called `protect`).
  This is useful:

  - To represent the quotation `'A` of `A`
  - To represent causality or direction:

  E.g.
  (superposition A (permute B))
  For 'B follows A'

  More generally, this can be used to encode sequences with a sumset (see below), by permuting the ith element ith times.

  - To randomize vectors (random permutation) (not sure how relevant with BSDC)"
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
  (comp dtt/->tensor f/+))

(def superposition bundle)

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
  ([a] (thin-pth-modulo a default-opts))
  ([a
    {:as opts
     :bsdc-seg/keys [segment-count segment-length N]}]
   (let [indices
         (-> (dtt/reshape a
                          [segment-count segment-length])
             (dtt/reduce-axis
              (fn [segment]
                (let [max-value (f/reduce-max segment)]
                  (when-not (zero? max-value)
                    (let [segment-max-indices
                          (dtype-argops/argfilter
                           (partial = max-value)
                           segment)
                          ;; 'p'
                          chosen-index
                          (segment-max-indices
                           (fm/mod
                            (long
                             (f/sum
                              segment-max-indices))
                            (count
                             segment-max-indices)))]
                      [chosen-index]))))
              -1
              :object))]
     indices
     (indices->hv* indices opts))))


(comment
  (let [a (->seed)] (= a (thin-pth-modulo a)))
  true
  (let [a (->seed)
        b (->seed)
        c (bundle a b)]
    [(maximally-sparse? a) (maximally-sparse? c)
     (maximally-sparse? (thin-pth-modulo c))
     (similarity c (thin-pth-modulo c))
     (similarity a (thin-pth-modulo c))
     (similarity b (thin-pth-modulo c))])
  [true false true 1.0 0.6 0.4])

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
;;
;;
;; For implemmentation of this on spiking neuromorphic hardware:
;; https://dl.acm.org/doi/fullHtml/10.1145/3546790.3546820
;;
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

  Trust me, you'll get a toaster out of it.

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


  This is communative and associative for maximally sparse vectors.
  (Obviously the case for all hdvs the [[bind*]] implementation.

  (require [bennischwerdtner.hd.binary-sparse-segmented :as hd])


  (= (hd/bind (clj->vsa :a) (clj->vsa :b))
     (hd/bind (clj->vsa :b) (clj->vsa :a)))
  true

  (= (hd/bind
      (clj->vsa :c)
      (hd/bind (clj->vsa :a)
               (clj->vsa :b)))
     (hd/bind (clj->vsa :b)
              (hd/bind
               (clj->vsa :c)
               (clj->vsa :a))))
  true




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

(defn bind*
  "
  Like [[bind]] but handles multiple inputs.

  This is slightly different from [[bind]] for non maximally sparse vectors,
  it returns a thinned result, where [[bind]] returns a rotated result.

  Arguably, this is a much simpler implementation.

  Zhonghao Yang 2023 (see lit.org)
  "
  ([hdvs] (bind* hdvs default-opts))
  ([hdvs {:bsdc-seg/keys [segment-count segment-length]}]
   (indices->hv (dtype/emap
                  #(fm/mod % segment-length)
                  :int8
                  (-> (dtt/reshape hdvs
                                   [(count hdvs)
                                    segment-count
                                    segment-length])
                      (dtt/reduce-axis dtype-argops/argmax)
                      (dtt/reduce-axis f/sum 0))))))


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

(defn unit-vector-n
  "
  Returns the `n` unit vector.

  This is the vector, that when bound [[bind]] with `a`, returns the same vector
  as rotating each segmente of a by n.


  (let [a (->hv)]
   (= (bind a (unit-vector-n 3)) (permute-block-local-n a 3)))
  => true

  (let [a (->hv)]
   (= (bind a (unit-vector-n 1)) (permute-inner a)))
  => true

  See [[permute-block-local-n]].
  "
  ([n] (unit-vector-n n default-opts))
  ([n
    {:as opts
     :bsdc-seg/keys [segment-count segment-length]}]
   (indices->hv (repeatedly segment-count
                            (constantly
                             (mod n segment-length)))
                opts)))




;; ---------------------------------------
;; This is an implementation of
;; Zhonghao Yang 2023, COGNITIVE MODELING AND LEARNING WITH SPARSE BINARY HYPERVECTORS
;; 'inverse bind' and 'unit vector'
;; See NOTICE
;;

(defn unit-vector
  "
  Returns the unit vector.

  The unit vector is identity element for `bind`:

  (let [a (->hv)]
    (= a (bind (unit-vector) a)))
  true

  "
  ([] (unit-vector default-opts))
  ([{:as opts :bsdc-seg/keys [segment-count]}]
   (unit-vector-n 0 opts)))

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



;; ---------------------------------------
;;
;; Zhonghao Yang 2023, COGNITIVE MODELING AND LEARNING WITH SPARSE BINARY HYPERVECTORS
;; See NOTICE
;;
;; Ends here
;;


;; I forget how this is called in the literature,
;; but we can drop bits from the vector.
;;
;; Similar to bundleling with random noise,
;; but making it thinner.
;;
;; Called it 'weaken' first.
;; Decided it should be `drop` for obviousness.
;;
;;
(defn drop
  "Returns a sparser version of `a` by dropping bits from it.

  `drop-ratio` 0 doesn't change anything:

  (let [a (->hv)]
    (= a (weaken a 0)))

  `drop-ratio` 1 returns a zero vector:

  (let [a (->hv)]
    (zero? (f/reduce-+ (weaken a 1))))

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

  Earlier alias: [[weaken]].
  "
  ([a drop-ratio] (drop a drop-ratio default-opts))
  ([a drop-ratio
    {:bsdc-seg/keys [segment-count segment-length N]
     :keys [tensor-opts]}]
   ;; the indices decide how to drop, is a form of
   ;; 'context dependent thinning'.
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
     (dtt/->tensor v tensor-opts))))

(def weaken drop)

(defn drop-randomly
  "Drop random bits from hv with `drop-chance`."
  ([hv drop-chance]
   (drop-randomly hv drop-chance default-opts))
  ([hv drop-chance opts]
   (dtt/->tensor
    (f/bit-and hv (->rand-mask (- 1 drop-chance) opts)))))



(comment

  )
