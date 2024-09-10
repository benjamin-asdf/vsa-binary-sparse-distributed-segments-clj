(ns bennischwerdtner.hd.data-next
  (:refer-clojure :exclude [replace set peek pop])
  (:require [tech.v3.datatype.functional :as f]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
            [tech.v3.datatype.bitmap :as bitmap]
            [fastmath.random :as fm.rand]
            [bennischwerdtner.pyutils :as pyutils :refer
             [*torch-device*]]
            [bennischwerdtner.hd.core :as hd]
            [tech.v3.datatype.unary-pred :as unary-pred]
            [tech.v3.datatype.argops :as dtype-argops]
            [bennischwerdtner.hd.impl.item-memory-torch :as
             item-memory-torch]
            [bennischwerdtner.hd.item-memory :as item-memory
             :refer
             [ItemMemory m-clj->vsa m-cleanup m-cleanup*
              m-cleanup-verbose]]))

(def ^:dynamic *item-memory*
  (item-memory/codebook-item-memory 1000))

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
       1 (hd/permute (clj->vsa* (first args)) (* dir 1))
       2 (hd/permute (clj->vsa* (first args)) (* dir (second args)))))))

(defn clj->vsa*-magic-unbind-impl
  [args]
  ;; action ⊙ source -> destination
  (let [[automaton a s d] args]
    (cond (= :_ a) (clj->vsa* [:. automaton s [:> d]])
          (= :_ s) (clj->vsa* [:. automaton a [:> d]])
          (= :_ d) (clj->vsa* [:< [:. automaton a s]]))))

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
        :* (apply hd/bind (map clj->vsa* (rest obj)))
        ;; should maybe be :/
        :.
        (hd/unbind
         (clj->vsa* (second obj))
         (apply
          hd/bind
          (map clj->vsa* (drop 2 obj))))
        :> (apply clj->vsa*-fish-impl obj)
        :< (apply clj->vsa*-fish-impl obj)
        :>> (apply clj->vsa*-fish-impl obj)
        :<< (apply clj->vsa*-fish-impl obj)
        :**>
        ;; [[transition]]
        (let [[a b c] (rest obj)]
          (hd/bind
           (clj->vsa* a)
           (clj->vsa* b)
           (hd/permute (clj->vsa* c))))
        :*.< (clj->vsa*-magic-unbind-impl (rest obj))
        ;; ~[[bound-seq]]
        :*> (apply hd/bind
                   (map-indexed (fn [i e] (hd/permute e i))
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




(comment
  (hd/similarity (clj->vsa :a) (clj->vsa :a))
  (hd/similarity (clj->vsa :a) (clj->vsa :b))
  (hd/similarity (clj->vsa :a) (clj->vsa :c))
  (cleanup
   (clj->vsa* [:. [:* :a :b] :a]))
  (cleanup
   (clj->vsa* [:+ :a :b]))

  (do
    (def *item-memory* (item-memory/codebook-item-memory 10))
    (cleanup*
     (clj->vsa* [:+ :a :b]))
    (cleanup-verbose
     (clj->vsa* [:+ :a :b])))
  (clj->vsa* [:* :a :b])

  (cleanup (clj->vsa* [:> :a]))
  (cleanup* (clj->vsa* [:< [:> :a 1]]))

  (cleanup*
   (clj->vsa*
    [:+ [:-- :a 0.5] [:-- :b 0.5]]))

  (cleanup* (clj->vsa* [:* :c [:+ [:-- :a 0.5] [:-- :b 0.5]]]))

  (clj->vsa*
   [:?=
    [:* :c [:+ [:-- :a 0.5] [:-- :b 0.5]]]
    :c])

  (clj->vsa*
   [:?= [:* :c [:+ [:-- :a 0.5] [:-- :b 0.5]]] :b])

  (cleanup*
   (clj->vsa*
    [:<
     [:.
      [:*> :a :b :c]
      :a
      [:> :b 1]]
     2]))

  (cleanup-verbose (clj->vsa* [:> :a]))
  (cleanup-verbose (clj->vsa* [:> :a]))

  (hd/similarity
   (hd/permute (clj->vsa* :a))
   (clj->vsa* [:> :a]))

  (hd/similarity
   (hd/permute (clj->vsa* :a))
   (clj->vsa* [:> :a])))



(comment
  (def a (hd/seed))
  (def b (hd/seed))
  (def c (hd/thin (hd/superposition a b)))
  (item-memory-torch/codebook-weights
   (item-memory-torch/->codebook-matrix [a b c])
   c)
  (item-memory-torch/codebook-weights
   (item-memory-torch/->codebook-matrix [a b c])
   b)
  ;; tensor([ 0., 20.,  8.], device='cuda:0',
  ;; dtype=torch.float16)
  (item-memory-torch/codebook-max
   (item-memory-torch/->codebook-matrix [a b c])
   a)
  (item-memory-torch/codebook-max
   (item-memory-torch/->codebook-matrix [a b c])
   c)
  (pyutils/ensure-jvm
   (:idxs (item-memory-torch/codebook-cleanup-verbose
           hd/default-opts
           (item-memory-torch/->codebook-matrix [a b c])
           c
           0.1))))
