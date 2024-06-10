(ns capacity
  (:require
   [bennischwerdtner.hd.binary-sparse-segmented :refer :all]
   [tech.v3.datatype.functional :as f]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [fastmath.random :as fm.rand]
   [fastmath.core :as fm]
   [tech.v3.datatype.argops :as dtype-argops]))


;; ======================
;; I. Bind Capacity
;; ======================

;; unbinding of bundled pairs
;; A VSA allows querying for a filler by unbinding the role.
;;
;; --
;; How many property-value (role-filler) pairs can be bundled and still provide the correct answer
;; to any query by unbinding a role?
;; --
;;

(defn bind-experiment
  [item-memory-n role-filler-pairs-k thin?]
  (let [ ;; n item-memory-n
        k role-filler-pairs-k
        item-memory (into []
                          (repeatedly item-memory-n ->hv))
        ;; now you make k pairs
        pairs (partition-all 2
                             (take (* 2 role-filler-pairs-k)
                                   (shuffle item-memory)))
        ;; each of these is like a Currency⊗Dollar
        ;; pair they must be binded and then bundled
        record (apply bundle
                      (map (fn [[a b]] (bind a b)) pairs))
        record (if thin? (thin record) record)
        cleanup-with-item-memory
        ;; I'm assuming when I get the correct
        ;; vector out of the item-memory, that
        ;; counts correct
        (fn [x]
          (last (sort-by #(similarity x %) item-memory)))
        ;; and now we check how many we can recover
        ;; It's like having a clojure map and asking
        ;; how many items I can assoc and recover. It
        ;; also works the other way around
        how-many-recovered
        (count (filter true?
                       (mapcat identity
                               (for [[k v] pairs]
                                 [(= (cleanup-with-item-memory
                                      (unbind record k))
                                     v)
                                  (= (cleanup-with-item-memory
                                      (unbind record v))
                                     k)]))))]
    (/ how-many-recovered (* 2 k))))


(bind-experiment 100 10 :thin)
1

(bind-experiment 1000 20 :thin)
1
9/10

;; (this might take minutes to calculate)

;; I guess the difference between me and Schlegel, Neuberg, Protzel 2021 is that they did not thin after bundling
(bind-experiment 1000 30 :thin)
23/30                                   ;  👈
;; without thinning:
(bind-experiment 1000 30 false)
1                                       ;  👈

(bind-experiment 1000 40 true)
57/80
(bind-experiment 1000 40 false)
1


;; with 50 kvps I only got rougly 0.5 capacity
(bind-experiment 1000 50 true)
11/20


;;
;; => This also agrees with Laiho et.al 2015,
;; it is possible and might be desireable to trade higher density for higher accuracy in places. 👈
;;


;; ====================
;; II. Bundle Capacity
;; ====================

;; This is asking how many vectors you can bundle and expect the result to still resemble the inputs.
;;

;; 1. Make an item memory of `n` random vectors -> `item-memory`
;; 2. Select `k` random vectors from the item-memory, without putting them back -> `vecs-for-bundle`
;; 3. Make a bundle of vecs-for-bundle, then either thin or not.
;; the resulting vector `query-vec` should be similar to all `vecs-for-bundle` and dissimilar to the rest of item memory
;; 4. Query `item-memory` with `query-vec`, take the top `k` vectors as result.
;; In the maximal query capacity case, these would be the `vecs-for-bundlex`.
;; Call the ratio of 'correctly retrieved' and 'uncorrectly retetrieved' vectors the `bundle capacity` (for n,k).


(defn bundle-experiment
  [n k thin?]
  (let [item-memory (into [] (repeatedly n ->hv))
        vecs-idx (take k (shuffle (range n)))
        vecs-for-bundle (map item-memory vecs-idx)
        query-vec (if thin?
                    (thin (apply bundle vecs-for-bundle))
                    (apply bundle vecs-for-bundle))]
    ;; now check how many of the k vecs are the top
    ;; similarity for the bundled vec
    (/ (count (clojure.set/intersection
               (into #{}
                     (take k
                           (sort-by
                            #(similarity % query-vec)
                            (fn [a b] (compare b a))
                            item-memory)))
               (into #{} vecs-for-bundle)))
       k)))

(into
 []
 (for [n (range 3)]
   (bundle-experiment 1000 20 true)))
[19/20 1 19/20]


(into
 []
 (for [n (range 3)]
   (bundle-experiment 1000 30 true)))
[13/15 4/5 13/15]


(f/mean
 (into
  []
  (for [n (range 10)]
    (bundle-experiment 1000 15 true))))
0.9933333333333334
;; -> with thinning a bundle capacity of 0.99+ is achieved for (n = 1000, k = 15)           👈
;; Bundleling with <= 15 is probably good for many use cases,
;;
;; If better capacity is needed, you can chose to use higher density for the query vector.
;;

;; Without thinning:
(into
 []
 (for [n (range 3)]
   (bundle-experiment 1000 20 false)))
[1 1 1]
