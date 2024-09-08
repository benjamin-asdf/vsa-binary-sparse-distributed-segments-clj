(ns bennischwerdtner.hd.data
  (:refer-clojure :exclude [replace set peek pop])
  (:require [tech.v3.datatype.functional :as f]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
            [tech.v3.datatype.bitmap :as bitmap]
            [fastmath.random :as fm.rand]
            [fastmath.core :as fm]
            [bennischwerdtner.hd.binary-sparse-segmented :as
             hd]
            [tech.v3.datatype.unary-pred :as unary-pred]
            [tech.v3.datatype.argops :as dtype-argops]
            [bennischwerdtner.hd.prot :refer
             [ItemMemory m-clj->vsa m-cleanup m-cleanup*
              m-cleanup-verbose]]))


;; Literature:
;;
;; [1]
;; Vector Symbolic Architectures as a Computing Framework for Emerging Hardware
;; Denis Kleyko, Mike Davies, E. Paxon Frady, Pentti Kanerva, Spencer J. Kent, Bruno A. Olshausen, Evgeny Osipov, Jan M. Rabaey, Dmitri A. Rachkovskij, Abbas Rahimi, Friedrich T. Sommer
;;
;; https://arxiv.org/abs/2106.05268
;;
;; [2]
;; Orthogonal Matrices for MBAT Vector Symbolic Architectures, and a "Soft" VSA Representation for JSON
;; https://arxiv.org/abs/2202.04771
;;
;; ----------------------
;;
;; See also:
;;
;; https://github.com/denkle/HDC-VSA_cookbook_tutorial
;;
;; -----------------------------------------



;; tiny item memory

(defrecord TinyItemMemory [m]
  ItemMemory
  (m-clj->vsa [this item]
    (or (@m item)
        ((swap! m assoc item (hd/->seed)) item)))
  (m-cleanup-verbose [this q threshold]
    (filter (comp #(<= threshold %) :similarity)
            (sort-by :similarity
                     #(compare %2 %1)
                     (into []
                           (pmap (fn [[k v]]
                                   {:k k
                                    :similarity
                                    (hd/similarity v q)
                                    :v v})
                                 @m)))))
  (m-cleanup-verbose [this q]
    ;; 0.18 checked around a little with segment-count = 20 this seems to be
    ;; very far apart
    ;; You quickly need cleanup memories for this stuff
    ;;
    (m-cleanup-verbose this q 0.18))
  (m-cleanup [this q] (first (m-cleanup* this q)))
  (m-cleanup* [this q]
    (map :k (m-cleanup-verbose this q))))

(def ^:dynamic *item-memory*
  (->TinyItemMemory (atom {})))

(defn clj->vsa [obj] (m-clj->vsa *item-memory* obj))
(defn cleanup-verbose
  ([q] (m-cleanup-verbose *item-memory* q))
  ([q threshold] (m-cleanup-verbose *item-memory* q threshold)))
(defn cleanup [q] (m-cleanup *item-memory* q))
(defn cleanup* [q] (m-cleanup* *item-memory* q))

(defn clj->vsa*-1
  [obj]
  (cond (hd/hv? obj) obj
        (set? obj) (apply hd/superposition
                          (map clj->vsa*-1 obj))
        (map? obj) (apply hd/superposition
                          (map (fn [[k v]]
                                 (hd/bind (clj->vsa*-1 k)
                                          (clj->vsa*-1 v)))
                               obj))
        (sequential? obj) (map clj->vsa*-1 obj)
        :else (clj->vsa obj)))


(declare clj->vsa*)

(defn clj->vsa*-fish-impl
  ([op & args]
   (let [dir ({:< -1 :<< -2 :> 1 :>> 2} op)]
     (case (count args)
       1 (hd/permute-n (clj->vsa* (first args)) (* dir 1))
       2 (hd/permute-n (clj->vsa* (first args)) (* dir (second args)))))))

(defn clj->vsa*-magic-unbind-impl
  [args]
  ;; action ⊙ source -> destination
  (let [[automaton a s d] args]
    (cond (= :_ a) (clj->vsa* [:. automaton s [:> d]])
          (= :_ s) (clj->vsa* [:. automaton a [:> d]])
          (= :_ d) (clj->vsa* [:< [:. automaton a s]]))))


;; potentially make a dsl like:
;;
(defn clj->vsa*
  "Returns hdvs according to clj->vsa* dsl.

  Not documented, you need to look at the code.
  This is experimental, use [[clj->vsa*-1]] if you don't want it.

  Look at the rich comment at the bottom of data.clj.

  "
  [obj]
  (cond
    ;; (= :_ obj) obj
    (vector? obj)
      (case (first obj)
        :+ (apply hd/superposition
             (map clj->vsa* (rest obj)))
        ;; :∪ :∩
        :- (let [a (clj->vsa* (nth obj 1))
                 sets (map (clj->vsa* (drop 2 obj)))]
             (dtt/->tensor
               (f/< 0 (f/- a (apply hd/superposition sets)))
               :datatype
               :int8))
        :* (hd/bind* (map clj->vsa* (rest obj)))
        ;; should maybe be :/
        :. (hd/unbind (clj->vsa* (second obj))
                      (hd/bind* (map clj->vsa*
                                  (drop 2 obj))))
        :> (apply clj->vsa*-fish-impl obj)
        :< (apply clj->vsa*-fish-impl obj)
        :>> (apply clj->vsa*-fish-impl obj)
        :<< (apply clj->vsa*-fish-impl obj)
        :**>
          ;; [[transition]]
          (let [[a b c] (rest obj)]
            (hd/bind* [(clj->vsa* a) (clj->vsa* b)
                       (hd/permute (clj->vsa* c))]))
        :*.< (clj->vsa*-magic-unbind-impl (rest obj))
        ;; ~[[bound-seq]]
        :*> (hd/bind* (map-indexed
                        (fn [i e] (hd/permute-n e i))
                        (map clj->vsa* (rest obj))))
        :?= (hd/similarity (clj->vsa* (nth obj 1))
                           (clj->vsa* (nth obj 2)))
        :?? (cleanup* (clj->vsa* (nth obj 1)))
        :-- (hd/drop (clj->vsa* (nth obj 1)) (nth obj 2))
        ;; :+> (clj->vsa* [:+ [:> (second obj)] (nth
        ;; obj 2)])
        (map clj->vsa* obj))
    (hd/hv? obj) obj
    (set? obj) (apply hd/superposition (map clj->vsa* obj))
    (map? obj) (apply hd/superposition
                 (map (fn [[k v]]
                        (hd/bind (clj->vsa* k)
                                 (clj->vsa* v)))
                   obj))
    (sequential? obj) (map clj->vsa* obj)
    :else (clj->vsa obj)))



;; ----------------
;; Sets
;; ----------------
;; An unorded composite of elements.
;;
;; Interestingly, this is shown to be equivalent to a bloom filter.
;; Just as with a bloom filter, a membership check is easy, just take the overlap [[hd/similarity]].
;; I.e. HDC/VSA is a superset of bloom filters, computing 'instantaneously'.
;;
;;

(defn set
  "Return a hdv that represents the *sumset* of the arguments.

  You might want to [[hd/thin]] the result."
  [& args]
  (apply hd/superposition args))

;; It works the same for multisets, too
(def multiset set)
(def union set)
;;
;; (... this disregard for what a whole and what are elemnts is typical for programing in superposition)
;;



(defn intersection-1
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
  ([sets] (intersection-1 (count sets) sets))
  ([threshold sets]
   (dtt/->tensor (f/<= threshold (apply union sets))
                 :datatype
                 :int8)))

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
  ([& sets] (intersection-1 sets)))

(defn difference
  "Returns the hdv reprenting the difference between `a` and `sets`.

  It is roughly all the elements that are in a, but not in `sets`.

  If `a` is a multiset, this makes the surving elements into a
  'single' set.

  If you want to compute with multisets, use [[f/-]] directly.
  "
  [a & sets]
  (dtt/->tensor (f/< 0
                     (f/- a (apply hd/superposition sets)))
                :datatype
                :int8))


;; ---------------
;; unit tests
;; ---------------
(comment
  (do (assert (= (hd/similarity (clj->vsa :b)
                                (intersection-1
                                 2
                                 [(f/+ (clj->vsa :a)
                                       (clj->vsa :b)
                                       (clj->vsa :c))
                                  (f/+ (clj->vsa :b)
                                       (clj->vsa 20))]))
                 1.0))
      (assert (= (hd/similarity (clj->vsa :a)
                                (intersection-1
                                  1
                                  [(f/+ (clj->vsa :a)
                                        (clj->vsa :b)
                                        (clj->vsa :c))
                                   (f/+ (clj->vsa :b)
                                        (clj->vsa 20))]))
                 1.0))
      (assert (< (hd/similarity (clj->vsa :c)
                                (intersection-1
                                  2
                                  [(f/+ (clj->vsa :a)
                                        (clj->vsa :b)
                                        (clj->vsa :c))
                                   (f/+ (clj->vsa :b)
                                        (clj->vsa 20))]))
                 0.1)
              (< (hd/similarity (clj->vsa :foo)
                                (intersection-1
                                  1
                                  [(f/+ (clj->vsa :a)
                                        (clj->vsa :b)
                                        (clj->vsa :c))
                                   (f/+ (clj->vsa :b)
                                        (clj->vsa 20))]))
                 0.1))
      (assert (= (hd/similarity (clj->vsa :b)
                                (intersection-1
                                  [(f/+ (clj->vsa :a)
                                        (clj->vsa :b)
                                        (clj->vsa :c))
                                   (f/+ (clj->vsa :b)
                                        (clj->vsa 20))]))
                 1.0))
      (assert (< (hd/similarity (clj->vsa :c)
                                (intersection-1
                                  [(f/+ (clj->vsa :a)
                                        (clj->vsa :b)
                                        (clj->vsa :c))
                                   (f/+ (clj->vsa :b)
                                        (clj->vsa 20))]))
                 0.1))))

;; -------------------------
;; Sequences
;; -------------------------
;;
;; - An ordered set of k elements [a,b,c,d,e], different from [e,b,a,c,d]
;; - Also called k-tuple,
;; - A pair is the special case where k = 2
;;
;; ------------------------
;; - This doesn't preserve similarity of nearby elements, there are proposals for that.
;; - The resultant hypervector has no information about predecessor or successor ship.
;;   proposal: https://pubmed.ncbi.nlm.nih.gov/21428993/
;;
;; [1] contains references to the literature.
;;


;; V1 - permute each element by p0,p1,p2,... then combine with superposition

(defn superpoisiton-seq
  "Returns an hdv that represents the sequence `xs`, which is a seq of hypervectors.

  Each data element of the sequence is permuted `i` times via [[hd/permute]],
  then we take the superposition of sequence elements.


  +----+     +----+     +----+
  | c  |     | b  |     | a  |    data elements
  +----+  ,  +----+  ,  +----+
    |          |          |
    | p0       | p1       | p2
    |          |          |
    v          v          v


  p0(c)  +   p1(b)   +   p2(a)   sequence elements

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

  Note that the first (last in `xs`) item is added as-is (p0(x) == x).
  The resulting hdv is similar to the first item.

  Note that this reverses the ordering of the items, like repeated conj or into would.

  "
  [xs]
  (apply hd/superposition
    (map-indexed (fn [i item] (hd/permute-n item i))
                 (reverse xs))))

(defn permute-seq-conj
  "Returns a new hdv representing the superpoisiton-seq with
  `e` added to the front of `hxs`."
  ([hxs e] (hd/superposition e (hd/permute hxs))))

(defn permute-seq-into
  "Returns a new hdv where `xs` elements are concatenated to the
  front of `hxs`, a hdv permute-sequence.

  This effectively reverses the order of `xs`, just as an iterative [[permute-seq-conj]] would.
  "
  ([hxs xs]
   (hd/superposition (hd/permute-n hxs (count xs))
                     (superpoisiton-seq xs))))

(defn permute-seq-nth
  "Returns a new hdv where the nth item of `hxs`, a 'permute seq'
  is at the front.

  Thereby making the return value similar to the `nth` item.
  Might also be called 'unmask index' or 'unquote index'.
  "
  ([hxs index] (hd/permute-n hxs (- index))))

;; ---------------
;; unit tests
;; ---------------
(comment
  (do
    ;; or by rotating the seq n times and probe
    ;; a perm-mapped seq is similar to its first elm
    (assert (= (cleanup (superpoisiton-seq (map clj->vsa
                                             [:a :b])))
               :b))
    ;; permute inverse 3 times and you are at :d
    (assert (= (cleanup (-> (superpoisiton-seq
                              (map clj->vsa
                                (reverse [:a :b :c :d :e])))
                            (hd/permute-n -3)))
               :d))
    ;; shifting stuff adding an item to the front is
    ;; easy
    (assert
      (= (let [hxs (superpoisiton-seq [(clj->vsa :b)
                                       (clj->vsa :a)])
               new-item (clj->vsa :0)
               new-hsx (hd/superposition new-item
                                         ;; shift
                                         ;; everything
                                         ;; one further
                                         (hd/permute hxs))]
           [;; double checking that new item was not
            ;; there initially
            (hd/similarity new-item hxs)
            (hd/similarity new-item new-hsx)
            ;; a is now at pos 2
            (hd/similarity (clj->vsa :a)
                           (hd/permute-inverse new-hsx))
            (= new-hsx (permute-seq-conj hxs new-item))])
         [0.0 1.0 1.0 true]))
    (assert (= (hd/similarity
                 (clj->vsa :y)
                 (permute-seq-into
                   (superpoisiton-seq [(clj->vsa :a)
                                       (clj->vsa :b)])
                   [(clj->vsa :x) (clj->vsa :y)]))
               1.0))
    (assert (= (cleanup* (permute-seq-into
                           (superpoisiton-seq
                             [(clj->vsa :a) (clj->vsa :b)])
                           [(clj->vsa :x) (clj->vsa :y)]))
               '(:y)))))

(comment
  (assert (= [(hd/similarity (clj->vsa :b)
                             (permute-seq-nth
                               (superpoisiton-seq (map clj->vsa
                                                [:a :b :c]))
                               1))
              (hd/similarity (clj->vsa :a)
                             (permute-seq-nth
                               (superpoisiton-seq [(clj->vsa :a)
                                               (clj->vsa :b)
                                               (clj->vsa
                                                 :c)])
                               1))]
             [1.0 0.0]))
  ;; nth with 0 is identity
  (assert (= (permute-seq-nth (superpoisiton-seq
                                [(clj->vsa :a) (clj->vsa :b)
                                 (clj->vsa :c)])
                              0)
             (superpoisiton-seq [(clj->vsa :a) (clj->vsa :b)
                             (clj->vsa :c)])))
  ;; nth with any number is not an error, but the
  ;; result is non-sense
  ;; (generally the case with HDC/VSA)
  (assert (nil? (cleanup (permute-seq-nth (superpoisiton-seq
                                            [(clj->vsa :a)
                                             (clj->vsa :b)
                                             (clj->vsa :c)])
                                          10)))))


;; V2 - permute like above but combine with [[hd/bind]].

;; 'chain' ?
(defn bound-seq
  "Like [[superpoisiton-seq]] but combines via [[hd/bind]], not superposition.

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


  I call this `chain` maybe, because the elements are all 'entangled',

  [[hd/unbind]] gives you the *other elements*; Given a subset of elements.
  "
  [xs]
  (hd/bind* (map-indexed (fn [i item] (hd/permute-n item i))
                         xs)))

(comment
  ;; similar to nothing
  (cleanup* (bound-seq (map clj->vsa [:a :b :c])))

  (assert
   (= (hd/permute (hd/bind (clj->vsa :a) (clj->vsa :b)))
      (hd/bind (hd/permute (clj->vsa :a)) (hd/permute (clj->vsa :b))))))


(defn bound-seq-conj
  "Returns a new hdv repsenting the bound seq with `e` added."
  [hxs e]
  ;; is the same as making a new tuple:
  ;; (bound-seq [e hxs])
  (hd/bind e (hd/permute hxs)))

(comment
  ;; The usage of a bound seq seems at first contrived,
  ;; but these constraints are useful for representing
  ;; things like trees
  ;; (below)
  ;;
  (assert (= (cleanup* (hd/permute-inverse
                        (hd/unbind (bound-seq (map clj->vsa
                                                   [:a :b]))
                                   (clj->vsa :a))))
             '(:b)))
  (assert (= (cleanup* (hd/unbind
                        (bound-seq (map clj->vsa [:a :b]))
                        (hd/permute (clj->vsa :b))))
             '(:a)))
  (assert (empty? (cleanup*
                   (hd/unbind
                    (bound-seq (map clj->vsa [:a :b :c]))
                    (hd/permute (clj->vsa :b)))))))



;;
;; Replacing Elements
;;

(defn superposition-seq-replace
  "Return a new hdv, representing the sequence where
  `old-e` at postion `i` is replaced with `new-e`.
  "
  [hxs i old-e new-e]
  (hd/superposition (f/- hxs (hd/permute-n old-e i))
                    (hd/permute-n new-e i)))

(defn bound-seq-replace
  "Return a new hdv, representing the sequence where
  `old-e` at postion `i` is replaced with `new-e`.
  "
  [hxs i old-e new-e]
  (hd/bind (hd/unbind hxs (hd/permute-n old-e i))
           (hd/permute-n new-e i)))

(comment
  (do (assert (nil? (cleanup (bound-seq (map clj->vsa
                                          [:a :b :c])))))
      (assert (= (cleanup
                   (hd/permute-n
                     (hd/unbind
                       (hd/unbind
                         (bound-seq (map clj->vsa
                                      [:a :b :c]))
                         (hd/permute-n (clj->vsa :a) 0))
                       (hd/permute-n (clj->vsa :b) 1))
                     -2))
                 :c))
      (assert (= (cleanup
                   (hd/permute-n
                     (hd/unbind
                       (hd/unbind
                         (bound-seq-replace (bound-seq
                                              (map clj->vsa
                                                [:a :b :c]))
                                            2
                                            (clj->vsa :c)
                                            (clj->vsa :d))
                         (hd/permute-n (clj->vsa :a) 0))
                       (hd/permute-n (clj->vsa :b) 1))
                     -2))
                 :d))))


;; ------------------

(defn permute-seq-select
  "Returns a hdv that this the superposition of the
  elements in `hxs` at the indices `indices`.

  + noise (as usual).
  "
  [hxs indices]
  (apply hd/superposition
    (map #(permute-seq-nth hxs %) indices)))

(defn permute-seq-take
  "Returns a hdv that is the superposition
  of the first `n` items in `hxs`.

  See [[permute-seq-select]]."
  [n hxs]
  (permute-seq-select hxs (range n)))

(comment
  (do
    (assert (= (let [subseq (permute-seq-select
                             (superpoisiton-seq
                              (map clj->vsa
                                   (reverse (range 5))))
                             [0 2 3])]
                 (into []
                       (comp (map #(hd/similarity % subseq))
                             (map #(< 0.1 %)))
                       (map clj->vsa (range 5))))
               [true false true true false]))
    (assert (= (let [subseq (permute-seq-nth
                             (superpoisiton-seq
                              (map clj->vsa
                                   (reverse (range 5))))
                             1)]
                 (into []
                       (comp (map #(hd/similarity % subseq))
                             (map #(< 0.1 %)))
                       (map clj->vsa (range 5))))
               [false true false false false]))))




;; --------------------------
;; Graphs
;; --------------------------
;;
;; - Graph (G) made from vertices and edges.
;; - Edges can be directed or undirected.
;;


;; --------------------------
;; 1. Make a random hyper vector for each vertex.
;; 2. An edge is represented by the bind of the two vertices.
;; 3. The whole graph G is the superposition of all edges.
;;
;; - this doesn't work for directed graphs
;;   (edges don't have a direction, bind is commutative)

(def ->undirected-edge
  "Does [[hd/bind]] on the arguments."
  (fn ([[a b]] (hd/bind a b)) ([a b] (hd/bind a b))))


;;
;; Saying that B follows A - it allows us to express causality.
;; (Could be called 'ergo' to honor Braitenbergs Ergotrix)
;; This is conceptually a cons cell.
;;
;; Quoting [2]:
;; Plate 2003 was aware of this issue and propposed several non-commutative
;; alternative formulations his HRR system, including matrix multiplication.
;;
;; Using permute seems to be a very elegant solution for having a non-commutative bind.
;;
(defn ->directed-edge
  "Returns a new hdv representing a directed edge from `a` to `b`.
  [[edge->source]], [[edge->destination]]."
  ([[a b]] (->directed-edge a b))
  ([a b] (hd/bind a (hd/permute b))))

(defn edge->source
  "Returns the `source` of a directed-edge `edge`,
  given a `query`, the `destination`."
  [edge destination]
  (hd/unbind edge (hd/permute destination)))

(defn edge->destination
  "Same as [[edge->source]], but `query` should be the `source`,
  and the outcome is the `destination`. "
  [edge source]
  (hd/permute-inverse (hd/unbind edge source)))

(defn undirected-graph
  "Returns a hdv that represents the undirected graph of `edges`.
  `edges` is a seq of pairs of vertices.

  This happens to be the same thing as a record or associative datastructure.
  "
  ([edges]
   (apply hd/superposition (map ->undirected-edge edges))))

(defn directed-graph
  "Returns a hdv that represents the directed graph of `edges`.

  `edges` is a seq of pairs of vertices.


  Limitations:

  - This doesn't represent vertices without any edges
  - You could either superpose the vertices, too. Or keep a second hdv
  with the set of vertices around.

  - For more discussion and usage see literature
    https://arxiv.org/abs/2106.05268
  "
  [edges]
  (apply hd/superposition (map ->directed-edge edges)))

(comment
  ;; saying that :e and :c are connected to :d
  (assert (= (cleanup* (hd/unbind (undirected-graph
                                   (map #(map clj->vsa %)
                                        [[:a :e]
                                         [:a :b]
                                         [:c :d]
                                         [:c :b]
                                         [:d :e]]))
                                  (clj->vsa :d)))
             '(:e :c)))

  (do
    ;; querying for edge membership
    (hd/similarity
     (->directed-edge (clj->vsa :a) (clj->vsa :b))
     (directed-graph
      (clj->vsa*
       [[:a :e] [:a :b] [:c :b] [:d :c]
        [:e :d]])))

    ;; asking what follows :d
    (assert (= (cleanup* (edge->destination
                          (directed-graph
                           (map #(map clj->vsa %)
                                [[:a :e] [:a :b] [:c :b]
                                 [:d :c] [:e :d]]))
                          (clj->vsa :d)))
               '(:c)))
    ;; and what goes into :d
    (assert (= (cleanup* (edge->source
                          (directed-graph
                           (map #(map clj->vsa %)
                                [[:a :e] [:a :b] [:c :b]
                                 [:d :c] [:e :d]]))
                          (clj->vsa :d)))
               '(:e)))

    ;; ...
    ;; You see something that is at first slightly strange, that is that the operations that look like
    ;; they work for elements work for the whole.
    ;;
    ;; `edge->source` you thought is only for an `edge`.
    ;; Not so in computing in superposition.
    ;; The whole responds to element operations. It's kinda like there is always allowed to be noise,
    ;; that the noise happens to be a datastructure for somebody else doesn't need to concern you.
    ;;
    ;; Same concept for when the sequence simply is similar as the first item, you can treat it as the first item if you want.
    ;;
    )

  ;; comparing graph similarity (edge membership) in superposition
  [(hd/similarity (directed-graph (map #(map clj->vsa %)
                                         [[:a :e] [:a :b] [:c :b]
                                          [:d :c] [:e :d]]))
                  (directed-graph (map #(map clj->vsa %)
                                         [ ;; [:a :e]
                                          ;; [:a :b]
                                          [:c :b] [:d :c]
                                          ;; [:e :d]
                                          ])))
   (hd/similarity (directed-graph (map #(map clj->vsa %)
                                         [[:a :e] [:a :b] [:c :b]
                                          [:d :c] [:e :d]]))
                  (directed-graph (map #(map clj->vsa %)
                                         [[:c :b] [:x :y]])))
   (hd/similarity (directed-graph (map #(map clj->vsa %)
                                         [[:a :e] [:a :b] [:c :b]
                                          [:d :c] [:e :d]]))
                  (directed-graph (map #(map clj->vsa %)
                                       [[:x :y] [:z :u]])))]
  [2.0 1.07 0.1]
  ;; similarity > 1.0 because it's dense (i.e. not maximally sparse) btw.

  )


;; ------------------------------------------
;; Binary Trees
;; ------------------------------------------
;; - Nodes have at most 2 childrend (left-child and right-child)
;; - trace: The path from root to leave
;; - leaves: terminal nodes
;;
;;
;; -----
;; Impl:
;;
;; The tree can be seen as a map where the keys are the traces like [:left :left :right] etc.
;;
;; trace:
;; right-right-left
;; At [1], they represent the trace with
;;
;; r ⊙ p(r) ⊙ p2(l)
;;

(def left-right-marker
  {:left (hd/->seed) :right (hd/->seed)})

;;
;; You can also pick a fresh left right for each depth
;; depends on the usage
;;
;; These are alternative ideas:
;; (def left-right-marker-depth
;;   (memoize (fn [idx left-right] (hd/->seed))))
;; (def depth-marker "Returns a well-known seed vector for `i`." (memoize (fn [i] (hd/->seed))))
;; (def left-right-marker {:left (hd/->seed) :right (hd/->seed)})
;;

(defn tree-trace*
  "Returns a tree trace representation.

  `trace` is a seq of hdvs."
  [trace]
  (bound-seq trace))

(defn tree*
  "Returns a tree representation,

  `trace-leave-pairs`:
  Each is of the form

  [[a b c] x]

  Where a,b,c and x are hdvs.

  This is the generalized form of [[tree]].
  a,b,c are allowed to be anything, so you can build 3-ary trees and so forth.

  It does not use the well known seed markers [[left-right-marker]].
  In case you want to compare trees build by [[tree]] and [[tree*]],
  you need to know that you are using [[left-right-marker]] hdvs.

  See [[tree-trace]], [[tree]].
  "
  [& trace-leave-pairs]
  (apply hd/superposition
         (map (fn [[trace leave]]
                (hd/bind (tree-trace* trace) leave))
              trace-leave-pairs)))

(defn tree-trace
  "Returns a tree trace representation.

  `trace` is a seq of :left or :right in symbolic domain."
  [trace]
  (tree-trace* (map left-right-marker trace)))

(defn tree
  "Returns a tree representation,

  `trace-leave-pairs`: Each a tuple of traces in symbolic domain and a leave hdv.

  Example:
   [[:left :left :left] :a]
   [[:left :right :left] :b])

  It is the superposition of key-value pairs where the keys are similar to what is produced by [[tree-trace]].
  And the values are leaves.

  See [[tree-trace]].
  "
  [& trace-leave-pairs]
  (tree* (map #(update % 0 tree-trace) trace-leave-pairs)))

(defn tree->leave
  "Returns a hdv that is similar to the leave of `tree` (a hdv),
  given a `trace` in symbolic domain.

  If your `trace` is in hyper domain, then use [[hd/unbind]].

  See [[tree-trace]].
  "
  [tree trace]
  (hd/unbind tree (tree-trace trace)))

(defn find-trace
  "Returns a hdv similar to the trace for tree, given a leave and tree."
  [tree leave]
  (hd/unbind tree leave))

(comment
  ;; tree capacity experiment

  (for
      [N [5 10 20 30 50 100 500 1000]]
      (let [tree-spec (into []
                            (for [n (range N)]
                              [(repeatedly (inc (rand-int 10))
                                           #(rand-nth [:left
                                                       :right]))
                               n]))
            tree (apply tree* (clj->vsa* tree-spec))]
        [:N N :mean
         (f/mean
          (for [n (range 20)]
            (let [tree-elm (rand-nth tree-spec)]
              (=
               (second tree-elm)
               (cleanup (hd/unbind
                         tree
                         (tree-trace*
                          (clj->vsa*
                           (first
                            tree-elm)))))))))]))
  '([:N 5 :mean 1.0]
    [:N 10 :mean 1.0]
    [:N 20 :mean 1.0]
    [:N 30 :mean 0.95]
    [:N 50 :mean 1.0]
    [:N 100 :mean 0.85]
    [:N 500 :mean 0.6]
    [:N 1000 :mean 0.0])


  ;; - build a tree with N random traces, of a random length between 1 and 10
  ;; - throw out duplicate traces, else the outcome is a superposition
  ;; - query 20 times and report the mean success rate of recovering the leaf
  ;;
  ;; this is with segment-count 20
  ;; -
  ;; - seems like tree with 15-20 elements should be fine
  ;;

  ;; if we where to thin we would trade speed for accuracy.
  ;; same thing with thinning already struggles with N = 5:

  (for [N [5 10 20 30 50 100]]
    (let [tree-spec (into []
                          (for [n (range N)]
                            [(repeatedly (inc (rand-int 10))
                                         #(rand-nth [:left
                                                     :right]))
                             n]))
          tree (hd/thin (apply tree* (clj->vsa* tree-spec)))]
      [:N N :mean
       (f/mean (for [n (range 20)]
                 (let [tree-elm (rand-nth tree-spec)]
                   (= (second tree-elm)
                      (cleanup (hd/unbind
                                tree
                                (tree-trace*
                                 (clj->vsa*
                                  (first
                                   tree-elm)))))))))]))

  '([:N 5 :mean 0.75]
    [:N 10 :mean 0.1]
    [:N 20 :mean 0.1]
    [:N 30 :mean 0.0]
    [:N 50 :mean 0.0]
    [:N 100 :mean 0.0]))








(comment
  ;; given trees a and b and a leave on a, you find
  ;; what 'the same leave' is for b
  (let [trace-b (find-trace (tree [[:left :left :left]
                                   (clj->vsa :a)]
                                  [[:left :right :left]
                                   (clj->vsa :b)])
                            (clj->vsa :b))
        tree2 (tree [[:left :left :left] (clj->vsa :a)]
                    [[:left :right :left] (clj->vsa :c)])]
    ;; ... in superposition
    (cleanup (hd/unbind tree2 trace-b)))
  ;; I thought this usage mode makes more sense.
  ;; clj->vsa* to go from symbolic to hd domain, tree*
  ;; a function in hd domain.
  (let [trace-b (find-trace
                  (apply tree*
                    (clj->vsa* [[[:left :left :left] :a]
                                [[:left :right :left] :b]]))
                  (clj->vsa :b))
        tree2 (apply tree*
                (clj->vsa* [[[:left :left :left] :a]
                            [[:left :right :left] :c]]))]
    (cleanup (hd/unbind tree2 trace-b))))




;; -----------------------------------
;;
;; Cleaning up a trace is harder
;; Here is an example, building a well known complete tree,
;; then this complete tree serves as template.
;;
;; Literature recommends resonator networks.
;;
;;
;; 1. given a leave and a tree, unbind -> resulting in hdv similar to the trace
;; 2. Query the well known tree with this trace, resulting in a well known leave,
;;    cleanup this leave -> symbolic domain
;; 3. (trivially) Map the cleaned up well known leave to a symbolic trace
;;
;;
(comment
  (def well-known-complete-tree
    ;; all 2^3 combinations of left right for depth 3
    (let [tree-spec (for [[idx trace]
                          (map-indexed
                           vector
                           (for [dir0 [:left :right]
                                 dir1 [:left :right]
                                 dir2 [:left :right]]
                             [dir0 dir1 dir2]))]
                      (let [well-known-leave (clj->vsa idx)]
                        [trace well-known-leave]))
          tree (apply tree tree-spec)]
      {:tree tree
       :tree-map (into {}
                       (map-indexed vector
                                    (map first tree-spec)))
       :tree-spec tree-spec}))
  (defn cleanup-trace
    [well-known-complete-tree tree leave]
    ((:tree-map well-known-complete-tree)
     (cleanup (hd/unbind (:tree well-known-complete-tree)
                         (hd/unbind tree leave)))))
  ;; -----------------------
  (cleanup-trace well-known-complete-tree
                 (tree [[:left :left :left] (clj->vsa :a)]
                       [[:left :right :left] (clj->vsa :b)]
                       [[:left :right :right]
                        (clj->vsa :c)])
                 (clj->vsa :b))
  [:left :right :left])

;; ...
;; The fact that a 'template' is a coherent concept shows that we are programing
;; with a substrate capable of representing analogies.
;; --------------------------------------

;;
;; Note that filler hypervectors could be any datastructure, doesn't have to be seed vectors.
;;


;; --------------
;; Unit Tests
;; -------------
(comment
  (assert
    (= [(cleanup*
          (hd/unbind
            (set (hd/bind (tree-trace [:left :left :left])
                          (clj->vsa :a))
                 (hd/bind (tree-trace [:left :left :right])
                          (clj->vsa :b)))
            (tree-trace [:left :left :right])))
        (cleanup*
          (tree->leave
            (set (hd/bind (tree-trace [:left :left :left])
                          (clj->vsa :a))
                 (hd/bind (tree-trace [:left :left :right])
                          (clj->vsa :b)))
            [:left :left :right]))]
       ['(:b) '(:b)]))
  (let [tree (tree [[:left :left :left] (clj->vsa :a)]
                   [[:left :right :left] (clj->vsa :b)])
        q (fn [query]
            (cleanup (hd/unbind tree (tree-trace query))))]
    (assert (= [(q [:left :right :left]) (q [:left :left])
                (q [:left :left :left]) (q [:right])
                (q [:right :left :left])]
               [:b nil :a nil nil]))))


(comment

  ;;
  ;; Alternative Tree
  ;; ----------
  ;; Use superposition for the trace representation
  ;;


  (defn tree-trace
    [trace]
    ;; instead of binding the trace, I use superposition
    ;; followed by thin.
    (hd/thin
     (apply
      hd/superposition
      (map-indexed
       (fn [idx lr]
         (left-right-marker-depth idx lr))
       (map left-right-marker trace)))))


  ;;
  ;; This representation as the property that the traces resemble each other
  ;;


  (hd/similarity
   (tree-trace [:left])
   (tree-trace [:left]))
  1.0

  (hd/similarity
   (tree-trace [:left])
   (tree-trace [:left :right]))
  0.5

  (hd/similarity
   (tree-trace [:left])
   (tree-trace [:left :right :left]))
  0.34

  (hd/similarity
   (tree-trace [:left :left])
   (tree-trace [:left :right :left]))
  0.14

  (hd/similarity
   (tree-trace [:left :right])
   (tree-trace [:left :right :left]))
  0.35

  (hd/similarity
   (tree-trace [:right :right])
   (tree-trace [:left :right :left]))
  0.16

  (hd/similarity
   (tree-trace [:right :right :left])
   (tree-trace [:left :right :left]))
  0.2

  ;; ----------------
  ;; Note:
  ;; It would be really cool to combine the upsides of a bind and a superposition trace representation.
  ;; If somehow the further to the root 2 traces differ, their traces would be more dissimilar.
  ;; ----------------

  ;;
  ;; This has the rather interesting effect that we can query for leaves where the traces resemble each other
  ;;

  (cleanup-verbose
   (hd/unbind (hd/superposition
               (hd/bind (clj->vsa :a)
                        (tree-trace [:left :left :left]))
               (hd/bind (clj->vsa :b)
                        (tree-trace [:left :right :left]))
               (hd/bind (clj->vsa :x)
                        (tree-trace [:left :left :right]))
               (hd/bind (clj->vsa :c)
                        (tree-trace [:right :left
                                     :right])))
              (tree-trace [:left :left :right])))



  '({:k :x :similarity 1.0}
    {:k :b :similarity 0.4}
    {:k :a :similarity 0.4}))


;; ------------------------
;; Stacks
;; ------------------------
;; - is almost the same as the superposition seq I have up top
;; - except that it should have pop impl that removes the item
;; - this requires cleaning up with item memory, which is not so easy, if
;;   you allow compound hdvs next to seed vectors
;; - (it would mean storing compounds into item memory, which would fill it up quickly)
;;
;;
;; I wonder why not just permute as a pop operation,
;;

(defn stack
  "Return a stack hdv with `items` added.

  Usage pattern for this is limited to ~7 items.
  "
  [& items]
  (apply hd/superposition
    (map-indexed (fn [i item] (hd/permute-n item i))
                 items)))

;; 'peek' silly?
(defn peek [stack] stack)

;;
;; [[clojure.core/pop]] returns the updated seq, so we do too
;; It's the only thing that makes sense with immutable datastructures
;;
(defn pop [stack]
  (hd/permute-inverse stack))

(defn pop-clean
  [stack cleanup]
  (let [clean (cleanup stack)]
    (hd/permute-inverse (f/- stack clean))))

(defn stack-conj [stack item]
  (hd/superposition item (hd/permute stack)))

(comment
  [(cleanup (hd/permute
              (pop-clean (stack (clj->vsa :a) (clj->vsa :b))
                         (fn [x] (clj->vsa (cleanup x))))))
   (cleanup (pop-clean (stack (clj->vsa :a) (clj->vsa :b))
                       (fn [x] (clj->vsa (cleanup x)))))]
  [nil :b])

;;
;; ... stack would need work to be useful for more than 7-9 items
;; at the moment, the stack would be to dense, it will resemble everything
;;
;; Maybe this is just fine, and the way to scale is with nested stuff and various item memories
;;







;; ----------------------------------------------------------

;; ---------------------------
;; Finite State Automata
;; ---------------------------
;;
;; A deterministic finite-state automaton is an abstract computational model.
;; - specify a set of states, a finite set of input symbols, a transition function,
;;   the start state, and a set of accepting states.
;; - Current state + input symbol determine next state
;; - Chaning a state is called transition
;; - the transition function all transitions in the automaton
;;
;;
;;


;; trunstile:

;; states:        { locked, unlocked }
;; input symbols: { token, push }
;;
;;
;; State diagram:
;; -------
;;
;;                                token (t)
;;                        +------------------+
;;                        |                  |
;;                 +------+--+         +-----v---+
;;                 | locked  |         | unlocked|
;;            +--->|   (l)   |         |  (u)    +----+
;;            |    +-+--^----+         +--+----^-+    |
;;            |      |  |                 |    |      |
;;            +------+  |                 |    +------+
;;   push(p)            +-----------------+        token (t)
;;                            push (p)
;;
;;



;;
;; - seed hdvs for states (you need an item memory for that)
;; - seed hdvs for input symbols
;;
;; the state diagram of is essentially a directed graph,
;; where each edge has an input symbol associated with it
;;
;; - so it's the same as the directed graph on top,
;;   but
;;
;;   vertex-a -> verxtex-b  needs an additional factor, the input symbol
;;
;;
;;   edge = token ⊙ left ⊙ permute(unlocked)
;;
;;
;; The superposition of the transitions represents the automaton (a)

;; written out that is:

(comment
  (let [turnstile
        (hd/superposition
         (hd/bind* [(clj->vsa :locked)
                    (hd/permute (clj->vsa :unlocked))
                    (clj->vsa :token)])
         (hd/bind* [(clj->vsa :locked)
                    (hd/permute (clj->vsa :locked))
                    (clj->vsa :push)])
         (hd/bind* [(clj->vsa :unlocked)
                    (hd/permute (clj->vsa :locked))
                    (clj->vsa :push)])
         (hd/bind* [(clj->vsa :unlocked)
                    (hd/permute (clj->vsa :unlocked))
                    (clj->vsa :token)]))
        ;; let's say I want to query for what is the next
        ;; state given
        ;; [:locked :token]
        ;; [current-state input-symbol] pair
        ;;
        ;; you see that the query becomes state ⊙ symbol
        ;; and the outcome is permuted once, hence
        ;; permute-inverse to get the outcome
        outcome (hd/permute-inverse
                 (hd/unbind turnstile
                            (hd/bind (clj->vsa :token)
                                     (clj->vsa :locked))))]
    (cleanup-verbose outcome)))

;; ({:k :unlocked
;;     :similarity 1.0
;;     :v #tech.v3.tensor<int8> [10000]
;;     [0 0 0 ... 0 0 0]})

;; (heheheh)

(defn transition
  "Returns an hdv representing a transition from `source` to `destination`,
  given `input`.

  Note that the roles of source and input can be swapped.
  `
  This is used as the element of [[finite-state-automaton]].
  "
  ([[source input destination]]
   (transition source input destination))
  ([source input destination]
   (hd/bind* [source input (hd/permute destination)])))

(defn finite-state-automaton-1
  "Creates a new finite state automaton containing `transitions`
  Also see [[finite-state-automaton]]"
  [transitions]
  (apply hd/superposition (map transition transitions)))

(defn finite-state-automaton
  "Returns an hdv representing (the transition function of),
  of a finite state automaton.

  `transitions` is a seq of tuples of the form

  [source input destination]



  -----------
  Example:
  -----------


  trunstile:
  -------------


   states:        { locked, unlocked }
   input symbols: { token, push }


   State diagram:
   -------


                                token (t)
                          +------------------+
                          |                  |
                   +------+--+         +-----v---+
                   | locked  |         | unlocked|
              +--->|   (l)   |         |  (u)    +----+
              |    +-+--^----+         +--+----^-+    |
              |      |  |                 |    |      |
              +------+  |                 |    +------+
     push(p)            +-----------------+        token (t)
                              push (p)



  You can create this via something like this:

  (apply
   finite-state-automaton
   (map #(map clj->vsa %)
      [[:locked :token :unlocked]
       [:locked :push :locked]
       [:unlocked :push :locked]
       [:unlocked :token :unlocked]]))

  Usage:

  (-> a
      (automaton-destination
       (clj->vsa :unlocked)
       (clj->vsa :token))
      cleanup)
  :unlocked

  See [[transition]], [[automaton-destination]], [[automaton-source]]
  "
  ([& transitions]
   (apply hd/superposition (map transition transitions))))

(defn fsa
  "Same as [[finite-state-automaton]] but `transitions` is a seq."
  ([transitions]
   (apply finite-state-automaton transitions)))

(defn automaton-destination
  "Returns a noisy hdv that is the result of
  querying `automaton`, a finite state automaton, for the next state,
  given `state` and `input-symbol`.
  "
  [automaton state input-symbol]
  (hd/permute-inverse
   (hd/unbind automaton (hd/bind state input-symbol))))

;; not sure about api
(defn automaton-source
  "

  Works for both source-state and input symbol.
  Technically, this is because bind is commutative.

  Source and input symbol contribute in the same way."
  [automaton source destination]
  (hd/unbind automaton
             (hd/bind source (hd/permute destination))))



(comment
  (assert
   (=
    (-> (apply finite-state-automaton
               (map #(map clj->vsa %)
                    ;; symbolic transition
                    [[:locked :token :unlocked]
                     [:locked :push :locked]
                     [:unlocked :push :locked]
                     [:unlocked :token :unlocked]]))
        (automaton-destination
         (clj->vsa :unlocked)
         (clj->vsa :token))
        cleanup)
    :unlocked))


  (-> (apply finite-state-automaton
             (map #(map clj->vsa %)
                  ;; symbolic transition
                  [[:locked :token :unlocked]
                   [:locked :push :locked]
                   [:unlocked :push :locked]
                   [:unlocked :token :unlocked]]))
      (automaton-destination
       (clj->vsa :unlocked)
       (clj->vsa :token))
      cleanup)
  :unlocked


  (let [turnstile (apply finite-state-automaton
                         (map #(map clj->vsa %)
                              ;; symbolic transition
                              [[:locked :token :unlocked]
                               [:locked :push :locked]
                               [:unlocked :push :locked]
                               [:unlocked :token :unlocked]]))]
    ;; let's say your query is "how do I unlock",
    ;; when you know it is currently locked
    ;;
    ;; this is intuitive:
    ;;
    ;; query = locked ⊙ p(unlocked)
    ;;
    ;; because the form
    ;;
    ;; `a` ⊙ p(`b`)
    ;;
    ;; is our general way of expressing b follows a.
    ;;
    ;; Saying 'I know that :unlocked follows :locked'
    ;; Saying 'this is the transition that I want to do,
    ;; what is the input for this?'.
    ;;
    (assert (= (cleanup (hd/unbind
                         turnstile
                         ;; what is the action that has the effect locked -> unlocked?
                         (hd/bind (clj->vsa :locked)
                                  (hd/permute (clj->vsa :unlocked)))))
               (cleanup (automaton-source turnstile
                                          (clj->vsa :locked)
                                          (clj->vsa :unlocked)))
               :token)))



  ;; ... and things sort of triangulate, if you know
  ;; two of speed, velocity or distance, you know the
  ;; other.
  ;;
  ;; And the same thing happens for
  ;; source-destination-symbol
  ;;
  ;;
  ;; I'm not going to draw triangles everywhere now,
  ;; but there is something deep about bind
  ;;
  ;; My current idea is that a cognitive system can
  ;; build
  ;;
  ;; [action state outcome] 'transitions', (~ Minsky
  ;; transframes)
  ;;
  ;; (the first 2 elements are interchangable, funny)
  ;;
  ;; when you know the destination and what you did
  ;; and you want to know where you came from:
  (let [turnstile (apply finite-state-automaton
                         (map #(map clj->vsa %)
                              ;; symbolic transition
                              [[:locked :token :unlocked]
                               [:locked :push :locked]
                               [:unlocked :push :locked]
                               [:unlocked :token :unlocked]]))]
    (assert
     (= (cleanup
         (automaton-source turnstile
                           (clj->vsa :token)
                           (clj->vsa :unlocked)))
        (cleanup (hd/unbind turnstile
                            (hd/bind (clj->vsa :token)
                                     (hd/permute
                                      (clj->vsa
                                       :unlocked)))))
        :locked))))

(comment

  ;; --------------------
  ;; Nondeterministic finite-state automaton
  ;; --------------------
  ;; - it can be in several states at once
  ;; - there can be several valid transitions from a given current state and input symbol
  ;; - It can assume a so-called generalized state,
  ;;   defined as a set of the automaton's states that are simultaneously active
  ;; - a generalized state corresponds to a hypervector representing the set of the currenlty active states
  ;; - query the same way, is like executing the automaton in parallel (in superposition)
  ;; - cleanup will have to search for several nearest neighbors
  ;;

  ;; automaton in superposition (i.e. just query with states that are in superposition)
  ;;

  (def water-domain
    (apply
     finite-state-automaton
     (clj->vsa*
      ;;  transition exressed in symbolic domain
      [[:frozen :heat :liquid] [:liquid :heat :gas]
       [:liquid :cool :frozen] [:gas :cool :liquid]
       [:gas :heat :gas] [:frozen :cool :frozen]])))

  (cleanup*
   (automaton-destination water-domain
                          (hd/superposition
                           (clj->vsa :liquid)
                           (clj->vsa :frozen))
                          (clj->vsa :cool)))
  '(:frozen)

  ;; if your state is the superposition of liquid and frozen

  (cleanup* (automaton-destination water-domain
                                   (hd/superposition
                                    (clj->vsa :liquid)
                                    (clj->vsa :frozen))
                                   (clj->vsa :heat)))
  '(:liquid :gas)

  ;; I mean, there is something else that is even crazier (or am I missing something?)
  ;; that is this:

  (def water-bender-domain
    (apply finite-state-automaton
           (map #(map clj->vsa %)
                [[:frozen :heat :shards]
                 [:liquid :heat :bubbles]
                 [:liquid :cool :absolute-zero]])))

  ;; now I have 2 automatons,

  (cleanup* (automaton-destination
             ;; ... superimpose them
             (hd/superposition water-domain water-bender-domain)
             (hd/superposition
              (clj->vsa :liquid)
              (clj->vsa :frozen))
             (clj->vsa :heat)))

  '(:liquid :gas :shards :bubbles)

  ;; and we just run them in parallel, lol
  ;; 'superposition' truly is the primitive means of combination of 'programming in superposition'
  ;;





  ;; I have chosen here the state and symbols to be the same though.
  ;; One would need to share or superimpose the symbols across domains
  ;; Either
  ;;
  ;; 1. have prototype roles for the symbols
  ;; 2. Find superpositions of symbol roles (perhaps using [[hd/thin]])
  ;;
  )


(comment
  (cleanup* (clj->vsa* [:+ :a :b]))
  '(:a :b)
  (cleanup* (clj->vsa* [:. [:* :a :b] :b]))
  (:a)
  (cleanup* (clj->vsa* [:. [:* :a :b :c] :b :c]))
  '(:a)
  (cleanup* (clj->vsa* [:< :b]))
  '()
  (cleanup* (clj->vsa* [:< [:> :b]]))
  '(:b)
  (cleanup* (clj->vsa* [:.
                        [:*> :a :b :c]
                        [:> :b]
                        [:> [:> :c]]]))
  '(:a)
  (cleanup* (clj->vsa* [:.
                        [:*> :a :b :c]
                        [:> :b]
                        [:>> :c]]))
  '(:a)
  (cleanup* (clj->vsa* [:<
                        [:+ [:*> :a :b :c] [:> :b]
                         [:> [:> :c]]]]))
  '(:b)
  (= (clj->vsa* [:* :c [:+ :a :b]])
     (clj->vsa* [:+ [:* :a :c] [:* :b :c]]))
  (hd/similarity (clj->vsa* [:+ [:* :a :c] [:* :b :c]])
                 (clj->vsa* [:* :c [:+ :a :b]]))
  1.0
  (cleanup* (clj->vsa* [:. [:* :c [:+ :a :b]] :c]))
  '(:a :b)
  (hd/similarity
   ;; (clj->vsa*
   ;;  [:+ [:* :a :c] [:* :b :c]])
   ;; (clj->vsa*
   ;;  [:* :c [:+ :a :b]])
   (hd/superposition
    (hd/bind* [(clj->vsa* :a) (clj->vsa* :c)])
    (hd/bind* [(clj->vsa* :b) (clj->vsa* :c)]))
   (hd/bind* [(clj->vsa* :c)
              (hd/superposition (clj->vsa* :a)
                                (clj->vsa* :b))]))
  1.0
  (hd/similarity (clj->vsa* [:+ [:* :a :c] [:* :b :c]])
                 ;; (clj->vsa*
                 ;;  [:* :c [:+ :a :b]])
                 (hd/bind* [(clj->vsa* :c)
                            (hd/superposition (clj->vsa* :a)
                                              (clj->vsa*
                                               :b))]))
  1.0
  (cleanup* (automaton-source (clj->vsa* [:**> :a :s :d])
                              (clj->vsa* :a)
                              (clj->vsa* :d)))
  '(:s)

  ;; [:*.< automaton :a :s :d]

  (cleanup* (clj->vsa* [:*.< [:**> :a :s :d] :_ :s :d]))
  '(:a)
  (cleanup* (clj->vsa* [:*.< [:**> :a :s :d] :a :_ :d]))
  '(:s)
  (cleanup* (clj->vsa* [:*.< [:**> :a :s :d] :a :s :_]))
  '(:d)

  (let [automaton [:+
                   [:**> :a :s :d]
                   [:**> :b :s :d]]]
    (cleanup* (clj->vsa* [:*.< automaton :_ :s :d])))
  '(:a :b)


  (let [automaton [:+
                   [:**> :a :s :d]
                   [:**> :b :s :d]]]
    (cleanup* (clj->vsa* [:*.< automaton :a :s :_])))
  '(:d)


  (let [automaton [:+
                   [:**> :a :s :d]
                   [:**> :b :s :d]]]
    (cleanup* (clj->vsa* [:*.< automaton (hd/drop (clj->vsa* :a) 0.5) :s :_])))
  '(:d)


  (let [tree-1
        (apply tree*
               (clj->vsa* [[[:left :left :left] :a]
                           [[:left :right :left] :b]]))
        tree-2 (clj->vsa*
                [:+
                 {[:*> :left :left :left] :a}
                 {[:*> :left :right :left] :b}])
        tree-3
        (clj->vsa*
         [:+
          [:* [:*> :left :left :left] :a]
          [:* [:*> :left :right :left] :b]])]
    (= tree-3 tree-2 tree-1))
  true

  ;; is all the same stuff of course:
  (let [a (hd/->seed)]
    (= (clj->vsa* :b)
       (hd/unbind (clj->vsa* [:* a :b]) a)))
  true

  (= (bound-seq-conj
      (bound-seq (clj->vsa* [:a :b :c]))
      (clj->vsa* :d))
     (clj->vsa* [:* :d [:> [:*> :a :b :c]]])
     (clj->vsa* [:* [:> [:*> :a :b :c]] :d]))
  true




  ;; well, whatever I guess
  ;; They would look cute but who can type it without emacs mode
  [:⊙ :a :b]
  [:⊘]
  [:⊕]
  ;; they look like happpy aliens with mouths
  ;;


  )


(comment

  ;; still mostly similar to b
  (hd/similarity
   (clj->vsa* :b)
   (clj->vsa*
    [:* :b (hd/drop (clj->vsa* :a) 0.75)]))
  0.8

  ;; so you can move something into another domain, but only a little bit.


  (clj->vsa*
   [:?= :b
    [:* :b [:-- :a 0.75]]])


  ;; unbinding with a gives me a little bit of b
  (clj->vsa*
   [:?= :b
    [:. [:* :b [:-- :a 0.75]] :a]])

  (let [coords {:bottom (clj->vsa* :bottom)
                :left (clj->vsa* :left)
                :right (clj->vsa* :right)
                :top (clj->vsa* :top)}]
    (let [[x y]
          ;; x, y
          [0 0]
          encode-position (fn [[x y]]
                            (clj->vsa*
                             [:+
                              ;; drop 0 (x) from top
                              [:-- :top y] [:-- :left x]
                              [:-- :right (- 1 x)]
                              [:-- :bottom (- 1 y)]]))
          ;;
          ;; the resultant encoding has 0 right, 0
          ;; bottom,
          ;; 1 top and 1 left
          ;; can be called 'top-left corner'
          field (clj->vsa*
                 [:+ [:* :banana (encode-position [0 0])]
                  [:* :orange (encode-position [0.5 0.5])]
                  [:* :triangle (encode-position [1 0.5])]
                  [:* :bloogy
                   (encode-position [(rand) (rand)])]
                  [:* :blerp
                   (encode-position [(rand) (rand)])]])]
      ;; where is the banana?
      ;; in the top left corner
      (map (fn [item] [item
                       (into {}
                             (map (fn [[k hv]] [k
                                                (clj->vsa*
                                                 [:?= hv
                                                  [:. field
                                                   item]])])
                                  coords))])
           [:banana :orange :triangle :square])))


  '(
    [:banana {:bottom 0.0 :left 0.6 :right 0.0 :top 0.45}]
    [:orange {:bottom 0.3 :left 0.2 :right 0.4 :top 0.25}]
    [:triangle {:bottom 0.2 :left 0.0 :right 0.7 :top 0.15}]
    [:square {:bottom 0.0 :left 0.05 :right 0.05 :top 0.0}])

  ;; square is nowhere, correct.
  ;;
  ;; triangle is only 0.15 top?
  ;;


  ;;
  ;; I feel like this must be approximately equal to population vectors
  ;;


  ;; other way around, ask what is at pos x

  (let [coords {:bottom (clj->vsa* :bottom)
                :left (clj->vsa* :left)
                :right (clj->vsa* :right)
                :top (clj->vsa* :top)}]
    (let [[x y]
          ;; x, y
          [0 0]
          encode-position (fn [[x y]]
                            (clj->vsa*
                             [:+
                              ;; drop 0 (x) from top
                              [:-- :top y] [:-- :left x]
                              [:-- :right (- 1 x)]
                              [:-- :bottom (- 1 y)]]))
          ;;
          ;; the resultant encoding has 0 right, 0
          ;; bottom,
          ;; 1 top and 1 left
          ;; can be called 'top-left corner'
          field (clj->vsa*
                 [:+ [:* :banana (encode-position [0 0])]
                  [:* :orange (encode-position [0.5 0.5])]
                  [:* :triangle (encode-position [1 0.5])]
                  [:* :bloogy
                   (encode-position [(rand) (rand)])]
                  [:* :blerp
                   (encode-position [(rand) (rand)])]])]
      ;; what is in the top left corner? They are all
      ;; sort of in the top left corner not bad, consider
      ;; that being on the field means your are a little
      ;; in the top-left corner
      ;;
      ;; the banana is the thing that is most in the top
      ;; left corner
      ;; (the output of cleanup* is sorted)
      [(clj->vsa* [:?? [:. field (encode-position [0 0])]])
       ;; what is in the center? Actually this also asks
       ;; 'what is in the field?'
       (clj->vsa* [:??
                   [:. field (encode-position [0.5 0.5])]])]))

  '[(:banana :blerp :orange :bloogy)
    (:orange :blerp :triangle :bloogy :banana)]

  ;; an orange and a blerp. Classic.
  ;; (is of course not deterministic, blerp and bloogy are all over the place)
  ;;




  ;; ---------------

  ;; The cogntive backdrop is the idea that 'eye movement motor data' 'bound' with 'sensor data'
  ;; encodes a visual field.
  ;;
  ;; Maybe you would find:
  ;; - eye movemnt population vectors encoding positions of objects in visual field
  ;; - Simultanagnosia, if any of the components is broken
  ;; - micro saccades whenever the sytem is representing the position of an object
  ;; - 1. micro saccades would come for the ride (although causility is circular) when paying attention to an obj.
  ;; - 2. perhaps there would be, depending on the amount of objects represented?

  )
