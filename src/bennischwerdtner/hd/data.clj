(ns bennischwerdtner.hd.data
  (:require
   [tech.v3.datatype.functional :as f]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [tech.v3.datatype.bitmap :as bitmap]
   [fastmath.random :as fm.rand]
   [fastmath.core :as fm]
   [bennischwerdtner.hd.binary-sparse-segmented :as
    hd]
   [tech.v3.datatype.unary-pred :as unary-pred]
   [tech.v3.datatype.argops :as dtype-argops]))


;; Literature:

;; Vector Symbolic Architectures as a Computing Framework for Emerging Hardware
;; Denis Kleyko, Mike Davies, E. Paxon Frady, Pentti Kanerva, Spencer J. Kent, Bruno A. Olshausen, Evgeny Osipov, Jan M. Rabaey, Dmitri A. Rachkovskij, Abbas Rahimi, Friedrich T. Sommer

;; https://arxiv.org/abs/2106.05268

;; See also:
;;
;; https://github.com/denkle/HDC-VSA_cookbook_tutorial


;; ----------------
;; Set
;; ----------------
;; An unorded composite of elements.
;;
;; Interestingly, this is shown to be equivalent to a bloom filter.
;; Just as with a bloom filter, a membership check is easy, just take the overlap [[hd/similarity]].
;; I.e. HDC/VSA is a superset of bloom filters, computing 'instantaneously'
;;
;;


(defn ->set
  "Return a hdv that represents the *sumset* of the arguments.

  You might want to [[hd/thin]] the result."
  [& args]
  (apply f/+ args))

(defn union "See [[->set]]." [sets] (apply f/+ sets))

(defn intersection
  "Return a hdv that represents the *intersection* between `sets`.

  `threshold`: The threshold for which an element counts as contributing
  to the intersection.

  If threshold is 1, then even a single bit contribution counts towards the intersection.
  If ommitted, this uses the count of `sets`, this works well when the input sets are normalized,
  (for instance by [[hd/thin]]).
  If all elements in sets are roughly seed vectors, where all contributing counters are 1.

  In other words, if `sets` are made from multisets, then a higher threshold might be needed.

  If threshold is 0, 'everything' (all ones) would be returned.
  "
  ([sets] (intersection (count sets) sets))
  ([threshold sets]
   (dtt/->tensor (f/<= threshold (union sets))
                 :datatype
                 :int8)))

;; It works the same for multisets, too
(def ->multiset ->set)


;; ---------------
;; unit tests
;; ---------------
(comment
  (let [m (atom {})]
    (defn clj->vsa
      [obj]
      (or (@m obj) ((swap! m assoc obj (hd/->seed)) obj))))
  (do
  (assert
    (= (hd/similarity
         (clj->vsa :b)
         (intersection
           2
           [(f/+ (clj->vsa :a) (clj->vsa :b) (clj->vsa :c))
            (f/+ (clj->vsa :b) (clj->vsa 20))]))
       1.0))
  (assert
    (= (hd/similarity
         (clj->vsa :a)
         (intersection
           1
           [(f/+ (clj->vsa :a) (clj->vsa :b) (clj->vsa :c))
            (f/+ (clj->vsa :b) (clj->vsa 20))]))
       1.0))
  (assert
    (= (hd/similarity
         (clj->vsa :c)
         (intersection
           2
           [(f/+ (clj->vsa :a) (clj->vsa :b) (clj->vsa :c))
            (f/+ (clj->vsa :b) (clj->vsa 20))]))
       0.0)
    (= (hd/similarity
         (clj->vsa :foo)
         (intersection
           1
           [(f/+ (clj->vsa :a) (clj->vsa :b) (clj->vsa :c))
            (f/+ (clj->vsa :b) (clj->vsa 20))]))
       0.0))
  (assert
    (= (hd/similarity
         (clj->vsa :b)
         (intersection
           [(f/+ (clj->vsa :a) (clj->vsa :b) (clj->vsa :c))
            (f/+ (clj->vsa :b) (clj->vsa 20))]))
       1.0))
  (assert
    (= (hd/similarity
         (clj->vsa :c)
         (intersection
           [(f/+ (clj->vsa :a) (clj->vsa :b) (clj->vsa :c))
            (f/+ (clj->vsa :b) (clj->vsa 20))]))
       0.0))))

(defn frequency
  "Returns a number that roughly corresponds to the *membership frequency* of `a` in `multiset`.

  `multiset` and `a` are both hdvs.

  This works when `a` is a seed vector (all non zero bits are 1's).
  Else, the outcome will be higher. The user is a consenting adult.
  "
  ([multiset a] (frequency multiset a hd/default-opts))
  ([multiset a {:bsdc-seg/keys [segment-count]}]
   (/ (f/dot-product multiset a) segment-count)))

(comment
  (assert (= (frequency (f/+ (clj->vsa :a)
                             (clj->vsa :a)
                             (clj->vsa :b)
                             (clj->vsa :c))
                        (clj->vsa :a))
             2.0))
  (assert (= (frequency (f/+ (clj->vsa :a)
                             (clj->vsa :a)
                             (clj->vsa :b)
                             (clj->vsa :c))
                        (clj->vsa :c))
             1.0)))



;; -------------------------
;; Sequence
;; -------------------------
;;
;; - An ordered set of k elements [a,b,c,d,e], different from [e,b,a,c,d]
;; - Also called k-tuple,
;; - A pair is the special case of k = 2
;;


;; V1 - permute each element by p0,p1,p2,... then combine with superposition

;; sequence-permutation-map
(defn ->permute-seq
  "Returns an hdv that represents the sequence `xs`, which is a seq of hypervectors.

  Each data element of the sequence is permuted `i` times via [[hd/permute]],
  then we take the superposition of sequence elements.


  +----+     +----+     +----+
  | a  |     | b  |     | c  |    data elements
  +----+  ,  +----+  ,  +----+
    |          |          |
    | p0       | p1       | p2
    |          |          |
    v          v          v


  p0(a)  +   p1(b)   +   p2(c)   sequence elements

    |          |          |
   -+----------+----------+ ∑
                          | superposition
                          v
                       +-----+
                       | hxs |  sequence hdv
                       +-----+

  This is equivalen to making a key-value pair mapping where the keys are
  [[hd/unit-vector-n]]'s.
  (Works like this here because permuting and binding are ~).

  Note that the first item is added as-is (p0(a) == a).
  The resulting hdv is similar to the first item.
  "
  ([xs] (->permute-seq xs hd/default-opts))
  ([xs opts]
   (apply hd/superposition
          (map-indexed (fn [i item] (hd/permute-n item i opts)) xs))))

(defn permute-seq-conj
  "Returns a new hdv representing the permute-seq with
  `e` added to the front of `hxs`."
  [hxs e]
  (hd/superposition e (hd/permute hxs)))

(defn permute-seq-into
  "Returns a new hdv where `xs` elements are concatenated to the
  front of `hxs`, a hdv permute-sequence.

  This effectively reverses the order of `xs`, just as an iterative [[permute-seq-conj]] would.
  "
  [hxs xs]
  (hd/superposition (hd/permute-n hxs (count xs))
                    (->permute-seq (reverse xs))))

(defn permute-seq-nth
  "Returns a new hdv where the nth item of `hxs`, a 'permute seq'
  is at the front.

  Thereby making the return value similar to the `nth` item.
  Might also be called 'unmask index' or 'unquote index'.
  "
  [hxs index]
  (hd/permute-n hxs (- index)))

;; ---------------
;; unit tests
;; ---------------
(comment
  (do
    ;; you can query a seq using [[hd/unit-vector-n]]
    (assert (= (hd/similarity
                (clj->vsa :b)
                (hd/unbind (->permute-seq [(clj->vsa :a)
                                           (clj->vsa :b)])
                           (hd/unit-vector-n 1)))
               1.0))
    ;; or by rotating the seq n times and probe
    ;; a perm-mapped seq is similar to its first elm
    (assert (= (hd/similarity (clj->vsa :a)
                              (->permute-seq
                               [(clj->vsa :a) (clj->vsa :b)
                                (clj->vsa :c) (clj->vsa :d)
                                (clj->vsa :e)]))
               1.0))
    ;; permute inverse 3 times and you are at :d
    (assert (= (hd/similarity
                (clj->vsa :d)
                (-> (->permute-seq
                     [(clj->vsa :a) (clj->vsa :b)
                      (clj->vsa :c) (clj->vsa :d)
                      (clj->vsa :e)])
                    (hd/permute-n -3)))
               1.0))
    (assert
     (= (hd/similarity (clj->vsa :b)
                       (hd/unbind (->permute-seq [(clj->vsa :a)
                                                  (clj->vsa
                                                   :b)])
                                  (hd/unit-vector-n 1)))
        1.0))
    ;; shifting stuff adding an item to the front is
    ;; easy
    (assert
     (= (let [hxs (->permute-seq [(clj->vsa :a)
                                  (clj->vsa :b)])
              new-item (clj->vsa :0)
              new-hsx (hd/superposition new-item
                                        ;; shift
                                        ;; everything
                                        ;; one further
                                        (hd/permute hxs))]
          [ ;; double checking that new item was not
           ;; there initially
           (hd/similarity new-item hxs)
           (hd/similarity new-item new-hsx)
           ;; a is now at pos 2
           (hd/similarity (clj->vsa :a)
                          (hd/unbind new-hsx
                                     (hd/unit-vector-n 1)))
           (= new-hsx (permute-seq-conj hxs new-item))])
        [0.0 1.0 1.0 true]))
    (assert (= (hd/similarity
                (clj->vsa :y)
                (permute-seq-into
                 (->permute-seq [(clj->vsa :a)
                                 (clj->vsa :b)])
                 [(clj->vsa :x) (clj->vsa :y)]))
               1.0))
    (assert (= (hd/similarity
                (clj->vsa :a)
                (permute-seq-into
                 (->permute-seq [(clj->vsa :a)
                                 (clj->vsa :b)])
                 [(clj->vsa :x) (clj->vsa :y)]))
               0.0))))


(comment
  (assert (=
            [(hd/similarity (clj->vsa :b)
                            (permute-seq-nth
                              (->permute-seq
                                [(clj->vsa :a) (clj->vsa :b)
                                 (clj->vsa :c)])
                              1))
             (hd/similarity (clj->vsa :a)
                            (permute-seq-nth
                              (->permute-seq
                                [(clj->vsa :a) (clj->vsa :b)
                                 (clj->vsa :c)])
                              1))]
            [1.0 0.0]))
  ;; nth with 0 is identity
  (assert (= (permute-seq-nth (->permute-seq
                                [(clj->vsa :a) (clj->vsa :b)
                                 (clj->vsa :c)])
                              0)
             (->permute-seq [(clj->vsa :a) (clj->vsa :b)
                             (clj->vsa :c)])))
  ;; nth with any number is not an error, the result is
  ;; non-sense
  ;; (generally the case with HDC/VSA)
  (assert (= (hd/similarity (clj->vsa :a)
                            (permute-seq-nth
                              (->permute-seq
                                [(clj->vsa :a) (clj->vsa :b)
                                 (clj->vsa :c)])
                              10))
             0.0)))


;; V2 - permute like above but combine with [[hd/bind]].

(defn ->bound-seq
  "Like [[->permute-seq]] but combines via [[hd/bind]], not superposition.

  Each data element of the sequence is permuted `i` times via [[hd/permute]],
  then we take the bind.


  +----+     +----+     +----+
  | a  |     | b  |     | c  |    data elements
  +----+  ,  +----+  ,  +----+
    |          |          |
    | p0       | p1       | p2
    |          |          |
    v          v          v


  p0(a)  +   p1(b)   +   p2(c)   sequence elements

    |          |          |
   -+----------+----------+ ∏
                          | bind
                          v
                       +-----+
                       | hxs |  sequence hdv
                       +-----+

  The resulting hdv is similar to *nothing* else, including no other hdv-sequences with similar items,
  except when the have the exact same order.

  "
  ([xs] (->bound-seq xs hd/default-opts))
  ([xs opts]
   (hd/bind* (map-indexed (fn [i item]
                            (hd/permute-n item i opts))
                          xs)
             opts)))



(comment
  (= (clj->vsa :a) (hd/permute-n (clj->vsa :a) 0))
  true







  (def a (hd/->seed))
  (= a (hd/bind a a))


  )
