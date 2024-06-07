(ns bennischwerdtner.hd.binary-sparse-segmented
  (:require [clojure.test :as t]
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

(defn maximally-sparse?
  "Returns true if `a` is maximally sparse.
  Then each segment normally has 1 non-zero bit."
  ([a] (maximally-sparse? a))
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
  ([a n] (permute-n n default-opts))
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
