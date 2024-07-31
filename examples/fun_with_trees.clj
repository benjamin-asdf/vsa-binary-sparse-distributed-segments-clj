(ns fun-with-trees
  (:require [tech.v3.datatype.functional :as f]
            [bennischwerdtner.hd.binary-sparse-segmented :as
             hd]
            [bennischwerdtner.hd.data :as hdd]))

;; ---------
;; Basics
;; -------

;; 1. Make a tree:

(def tree1
  (apply hdd/tree*
    (hdd/clj->vsa* [[[:left :left :left] :a]
                    [[:left :right :left] :c]
                    [[:left :right :right] :d]
                    [[:left :left :right] :e]])))

;;
;; It's essentially a mapping from trace "chains" to leaves.
;;

;; I leave the tree dense
(hd/maximally-sparse? tree1)
false
;; It's the superposition of it's elements. The element is a trace-leave pair.
;; You can [[hd/thin]] but you trade accuracy fast.



;; -----------
;; 2. Query a leave given a trace:

(let [trace (hdd/tree-trace* (hdd/clj->vsa* [:left :right :left]))]
  (hdd/cleanup (hd/unbind tree1 trace)))
:c

;; -----------
;; 3. Given a second tree, find the 'analogous' leave element
;; (this is very similar to 'what is the dolar in mexico?'.
;;

(def tree2
  (apply hdd/tree*
    (hdd/clj->vsa* [[[:left :left :left] :foo]
                    [[:left :right :left] :bar]])))


(hdd/cleanup
 (let [query
       (hd/unbind tree2 (hdd/clj->vsa* :foo))]
   (hd/unbind tree1 query)))
:a


;; how does this work?

;;
;; In order to reason through programing in superposition,
;; one trick is to simply not think about the whole, but about a single element.
;; The rest comes along for the ride.
;;

;; Tree is made from 'chain' elements, a bound (⊙) sequence, or 'key value pairs',
;; where the key is a trace and the value is the leave
;;

;;
;;      +--------------+     +-----+
;;      |   trace      | ⊙  |leave|
;;      +--------------+     +-----+
;;
;;
;;

;; We have 2 trees, which contain these elements:
;;
;;
;;
;;      +--------------+     +-----+
;;      |   trace      | ⊙  |  a  |     Tree  1
;;      +--------------+     +-----+
;;
;;
;;      +--------------+     +-----+
;;      |   trace      | ⊙  | foo  |    Tree  2
;;      +--------------+     +-----+
;;


;;
;; Algorithm:
;;
;;
;;  1. Approach Tree 2 with the leave query:
;;
;;
;;
;;      +--------------+     +-----+
;;      |   trace      | ⊙  | foo  |    Tree  2
;;      +--------------+     +-----+
;;              +--------------+
;;              |         ⊘   |
;;              |              |
;;              |
;;              |              foo
;;              |
;;              v
;;
;;            trace
;;
;; ⊘ means unbind
;;
;;
;;  The resultant hdv is `trace` (or similar to trace, given noise).
;;
;;
;;
;;  2. Approach Tree 1 with the resultant hdv (~ `trace` )
;;
;;
;;      +--------------+     +-----+
;;      |   trace      | ⊙  |  a  |     Tree  1
;;      +--------------+     +-----+
;;           +------------------+
;;           |       ⊘         |
;;           |                  |
;;           |                  |
;;           |                  |
;;        ~trace                |
;;                              |
;;                              v
;;                              a + noise
;;


;;
;; The whole superposition business does nothing but add some noise.
;; Cleaning up the result results in `a`.
;;

;; ---------------------------
;; Superposition Basics
;; ---------------------------
;;
;; This was cool, but here begins the superposition ride.
;;
;;

;; 1. Query in superposition
;; -------------------------
;;
;; querying with a 'skeleton tree' results in the superposition of the leaves.
;;
;;
;; Consider:
;;
;;
;; superposition (  `trace-1`, `trace-2`, `trace-3` )
;;
;;
;; This represents either a 'query in superposition', or a 'skeleton tree' of these 3 traces.
;; (I.e. it is a tree without leaves)
;; (that is equivalent)
;;

(def skeleton-tree
  (hd/superposition
   (hdd/tree-trace* (hdd/clj->vsa* [:left :right :left]))
   (hdd/tree-trace* (hdd/clj->vsa* [:left :left :right]))
   (hdd/tree-trace* (hdd/clj->vsa* [:left :right :right]))))

;; (this is also equivallent to a tree with 'zero unit vector' as leaves)
(= skeleton-tree
   (hd/superposition
    (hd/bind (hdd/tree-trace* (hdd/clj->vsa* [:left :right :left])) (hd/unit-vector-n 0))
    (hd/bind (hdd/tree-trace* (hdd/clj->vsa* [:left :left :right])) (hd/unit-vector-n 0) )
    (hd/bind (hdd/tree-trace* (hdd/clj->vsa* [:left :right :right])) (hd/unit-vector-n 0))))
true


(hdd/cleanup* (hd/unbind tree1 skeleton-tree))
'(:c :d :e)

;; lol, the outcome are the 'selected' leaves in superposition


;;
;; 2. Analogous elements
;; -----------------------------
;;
;; Things kinda go in multiple directions in this way:

;; Let's say we have a superposition of leaves at hand,
;;
(def leaves-q
  (hd/superposition
   (hdd/clj->vsa* :foo)
   (hdd/clj->vsa* :bar)))

;; make a query, get the 'trace structure' (or the skeleton tree representing the paths to our leaves).

(hd/unbind tree1 leaves-q)

;; ... is of course the superposition of the contributing traces.
;; Then you ask another tree 'what is analogous?':
;;
;; -----------
;; For implementation, we need to recover cleaned up traces, else the noise is too high.
;; This can be done with an item memory. 'Resonator networks' (see literature) have been suggested to this this efficiently.
;;

;; -------------
;; tree traces bookkeeping
(def trace-item-memory (atom #{}))

(defn cleanup-trace*
  [hdv]
  (->> (pmap (fn [e]
               {:e e :similarity (hd/similarity e hdv)})
             @trace-item-memory)
       (filter (comp #(<= 0.1 %) :similarity))
       (map :e)))

(defn remember-tree-traces
  [trace-pairs]
  (swap! trace-item-memory clojure.set/union
    (into #{}
          (map hdd/tree-trace* (map first trace-pairs))))
  trace-pairs)
;; -----------------


(let [query1 (hd/unbind (apply hdd/tree*
                               (remember-tree-traces
                                (hdd/clj->vsa*
                                 [[[:left :left :left] :foo]
                                  [[:left :right :left] :bar]
                                  [[:left :right :right] :bar2]
                                  [[:left :left :right]
                                   :bar3]])))
                        ;; leaves-q
                        (hd/superposition (hdd/clj->vsa* :foo) (hdd/clj->vsa* :bar))
                        )
      ;;
      ;; query1 is sort of the skeleton representing the 2 traces going to :foo and :bar.
      ;; But query1 is too dirty to work, so clean it up.
      cleaned-up-query (apply hd/superposition
                              (cleanup-trace* query1))]
  ;; this is just to check that the cleanup memory worked like we wanted.
  [:cleaned-up-query-is-equal-to-skeleton-tree
   (= cleaned-up-query
      (hd/superposition
       (hdd/tree-trace* (hdd/clj->vsa* [:left :left :left]))
       (hdd/tree-trace* (hdd/clj->vsa* [:left :right
                                        :left]))))
   :query-analogous-leaves
   (hdd/cleanup-verbose
    (hd/unbind (apply hdd/tree*
                      (hdd/clj->vsa*
                       [[[:left :left :left] :lolfoo]
                        [[:left :right :left] :lolbar]
                        [[:right :right :right] :hurr]
                        [[:right :right :left] :lolfoohehe]
                        [[:left :right :right] :bar2]]))
               cleaned-up-query))])

#_[
   :cleaned-up-query-is-equal-to-skeleton-tree true

   :query-analogous-leaves
   ({:k :lolfoo
     :similarity 0.5
     :v #tech.v3.tensor<int8> [10000]
     [0 0 0 ... 0 0 0]}
    {:k :lolbar
     :similarity 0.5
     :v #tech.v3.tensor<int8> [10000]
     [0 0 0 ... 0 0 0]})]


;; adding more noise to query vector:

(for [n (range 10)]
  (let [query1 (hd/unbind
                 (apply hdd/tree*
                   (remember-tree-traces
                     (hdd/clj->vsa*
                       [[[:left :left :left] :foo]
                        [[:left :right :left] :bar]
                        [[:left :right :right] :bar2]
                        [[:left :left :right] :bar3]])))
                 (hd/superposition
                   (hdd/clj->vsa* :foo)
                   ;; with some noise it
                   ;; starts failing
                   ;; sometimes
                   (hd/->hv)
                   (hd/->hv)
                   (hd/->hv)
                   ;; with 6 contributing
                   ;; vectors it starts
                   ;; failing ocasionally
                   (hdd/clj->vsa* :bar2)
                   (hdd/clj->vsa* :bar)))
        cleaned-up-query (apply hd/superposition
                           (cleanup-trace* query1))]
    (hdd/cleanup* (hd/unbind
                    (apply hdd/tree*
                      (hdd/clj->vsa*
                        [[[:left :left :left] :lolfoo]
                         [[:left :right :left] :lolbar]
                         [[:right :right :right] :hurr]
                         [[:right :right :left] :lolfoohehe]
                         [[:left :right :right] :bar2]]))
                    cleaned-up-query))))

'((:lolfoo :bar2 :lolbar)
 (:lolfoo :bar2 :lolbar)
 (:lolfoo :bar2 :lolbar)
 (:lolfoo :bar2 :lolbar)
 (:lolfoo :bar2 :lolbar)
 (:lolfoo :bar2 :lolbar)
 (:bar2 :lolbar)
 (:lolfoo :bar2 :lolbar)
 (:lolfoo :bar2 :lolbar)
 (:bar2 :lolbar))

;; 8/10 times in this config we recover the 'analogous' leaves from the tree at hand.
;; So here is a limitation of this version.
;;


;; ----------------------------
;; Superposition - similarity
;; ----------------------------
;;
;; Datastructures in superposition generally give a similarity judgement via their overlap.
;; [[hd/similarity]] looks for the overlap and normalizes with maximal sparsity,
;; i.e. more than 2.0 similarity means 2 'seed vectors worth of overlap'.
;; (2 seed vectors can maximally contribute 1.0 similarity).
;;

(def tree1
  (apply hdd/tree*
         (hdd/clj->vsa* [[[:left :left :left] :a]
                         [[:left :right :left] :c]
                         [[:left :right :right] :d]
                         [[:left :left :right] :e]])))

(hd/similarity
 tree1
 (apply hdd/tree* (hdd/clj->vsa* [[[:left :left :left] :a]])))
1.0

(hd/similarity
 tree1
 (apply hdd/tree*
        (hdd/clj->vsa* [[[:left :left :left] :b]])))
0.0


(hd/similarity
 tree1
 (apply hdd/tree*
        (hdd/clj->vsa* [[[:left :left :left] :a]
                        [[:left :right :left] :c]])))
2.0

(hd/similarity
 tree1
 (apply hdd/tree*
        (hdd/clj->vsa* [[[:left :left :left] :a]
                        [[:left :right :left] :c]
                        [[:left :right :right] :e]])))
2.0

;; ----------------------------
;; Everything is a Set
;; ----------------------------
;;
;; - superposition datastructures are sets of elements
;;

;; In some ways, the set is the most primitive superposition datastructure,
;; you see this by it being a synonym for  [[hd/superposition]]

(=
 (hd/superposition (hdd/clj->vsa :a) (hdd/clj->vsa :b))
 (hdd/set (hdd/clj->vsa :a) (hdd/clj->vsa :b)))
true

;; Union, intersection, difference work for all superimposed datastructures
;; ---------------------------------------------------------------------

;; Basics:
;;
;; Trees are superpositions of elements. Saying union is just like adding more elements.
;;

(hdd/cleanup
  (let [combinend-tree
          (hdd/union (apply hdd/tree*
                       (hdd/clj->vsa*
                         [[[:left :left :left] :a]
                          [[:left :left :right] :b]]))
                     (apply hdd/tree*
                       (hdd/clj->vsa* [[[:right :left
                                         :right] :x]])))]
    (hd/unbind combinend-tree
               (hdd/tree-trace* (hdd/clj->vsa* [:right :left
                                                :right])))))
:x


;;
;; If you 'union', you get the union, structurally.
;;
;; "Union superimposes structurally"
;; ----------------------------------

;; Consider 2 trees with an overlap trace (trace1) [:left :left :left] in example below.
;;

;;
;;                     tree 1                    tree 2
;;                     +----+                    +----+
;;                     |    |                    |    |
;;                     +-|--+                    +-+--+
;;                       |  | |                    |
;;               +-------+  | |                +---+
;;      trace1   |          v v                |
;;               |        (+other traces)      |
;;               v         ~ noise             v
;;              :a                             :x
;;
;;
;; When we take the union of these 2:
;;
;; (union tree1 tree1) -> combinend-tree
;;
;; combinend-tree looks like this:
;;
;;
;;                     combinend-tree
;;                     +----+
;;                     |    |
;;                     +-|--+
;;                       |  | |
;;               +-------+  | |
;;      trace1   |          v v
;;               |        (+other traces from tree1)
;;               v         ~ noise
;;           (⊕ :a :x)
;;
;;
;; ⊕ means superposition
;;


(hdd/cleanup*
 (let [combinend-tree
       (hdd/union
        (apply hdd/tree*
               (hdd/clj->vsa* [[[:left :left :left] :a]
                               [[:left :left :right] :b]]))
        (apply hdd/tree*
               (hdd/clj->vsa* [[[:left :left :left] :x]])))]
   (hd/unbind combinend-tree
              (hdd/tree-trace* (hdd/clj->vsa* [:left :left
                                               :left])))))
'(:a :x)


;; How does this work?
;;
;; The mathematical answer is that binding distributes over superposition:
;;
;;

;;   (⊕ (⊙ :a :b) (⊙ :a :c))
;; = (⊙ :a (⊕ :b :c))

(= (hd/superposition
    (hd/bind (hdd/clj->vsa :a) (hdd/clj->vsa :b))
    (hd/bind (hdd/clj->vsa :a) (hdd/clj->vsa :c)))
   ;; --------------
   (hd/bind
    (hdd/clj->vsa :a)
    (hd/superposition (hdd/clj->vsa :b)
                      (hdd/clj->vsa :c))))
true

;;
;; replacing :a for 'trace' (and treating tree as an element + noice)
;; yields the situation up top
;;
;;

;; ---------
;; Another way to put it is to say that binding with a maps the points into the ":a domain".
;;
;; Then, in the ":a domain", you can ⊕ your points, creating a similar point between the two.
;;
;; Since binding preserves similarity, this mapping can be done in any order, I can also map an
;; already packaged point into the ":a domain."
;;
;; ---------
;; To elaborate on that theme, it is like the tree is a collection of little 'trace domains',
;; if you superimpose a collection of points, the points that are 'inside' one of the trace domains
;; are in a superposition relationship.
;; (this doesn't mean tha the points going in are similar, they are not. The superpoosition coming out is
;; similar to both)
;; ----------
;;
;; How does *that* work?
;;
;;


;; Intersection works structurally
;; -------------------

;; This is essentially coming from the same properties
;;
;; For a tree, this means that only the trace-leave pairs that are shared will be part of the intersection:

(let [intersect-tree
      (hdd/intersection
       [(apply hdd/tree*
               (hdd/clj->vsa* [[[:left :left :left] :a]
                               [[:left :left :right] :b]]))
        (apply hdd/tree*
               (hdd/clj->vsa*
                ;; note, this is the shared
                ;; trace-leave pair
                [[[:left :left :left] :a]
                 [[:left :right :right] :x]]))])]
  [
   ;; a?
   (hdd/cleanup (hd/unbind intersect-tree (hdd/tree-trace* (hdd/clj->vsa* [:left :left :left]))))
   ;; b?
   (hdd/cleanup (hd/unbind intersect-tree (hdd/tree-trace* (hdd/clj->vsa* [:left :left :right]))))])
[:a nil]
;; :b was cut out
;;


;; Difference works structurally
;; -------------------

(let [difference-tree
        (hdd/difference
         (apply hdd/tree*
                (hdd/clj->vsa* [[[:left :left :left] :a]
                                [[:left :left :right] :b]]))
         (apply hdd/tree*
                (hdd/clj->vsa*
                 ;; note, this is the shared
                 ;; trace-leave pair
                 [[[:left :left :left] :a]
                  [[:left :right :right] :x]])))]
  [;; a?
   (hdd/cleanup (hd/unbind difference-tree
                           (hdd/tree-trace* (hdd/clj->vsa*
                                              [:left :left
                                               :left]))))
   ;; b?
   (hdd/cleanup (hd/unbind difference-tree
                           (hdd/tree-trace* (hdd/clj->vsa*
                                              [:left :left
                                               :right]))))])
[nil :b]
;; flipped with respect to above.
;; This time :b was leftover
;;




;; The intersection with random noise is nothing
;; -------------------------------------------------
;;

(f/mean
 (for [n (range 100)]
   (f/sum
    (hdd/intersection
     [tree1 (hd/->hv)]))))
0.16

;;
;; The overlap between tree1 and random noise (->hv) is 1-2 bits on average
;; (N = 10.000 and segment-count = 20)
;; (the resultant hypervector is ~ nothing)
;;
;; (`nothing` is the 'zero' hdv).
;;
