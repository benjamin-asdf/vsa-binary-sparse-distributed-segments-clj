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
            [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python.ffi :as py-ffi]
            [libpython-clj2.python :refer [py. py.. py.-]
             :as py]))

(defn cleanup-idx
  ([mem x] (cleanup-idx mem x 0.18))
  ([mem x threshold]
   (let [scores (hd/similarity mem x)
         [value index] (into [] (torch/max scores :dim -1))]
     (when (<= threshold (py.. value item))
       index))))

(defn cleanup-1
  ([mem x] (cleanup-1 mem x 0.18))
  ([mem x threshold]
   (some->> (cleanup-idx mem x threshold)
            (py/get-item mem))))

(defn cleanup-verbose-1
  ([mem x] (cleanup-verbose-1 mem x 0.18))
  ([mem x threshold]
   (let [scores (hd/similarity mem x)
         [value index] (into [] (torch/max scores :dim -1))]
     (when (<= threshold (py.. value item))
       {:idx index
        :item (torch/index_select mem -2 index)
        :sim value}))))

(defn cleanup*-1
  [mem x threshold]
  (let [scores (hd/similarity mem x)]
    (py/get-item mem (torch/ge scores threshold))))


;; (defn cleanup-verbose*
;;   ([mem x] (cleanup-verbose-1 mem x 0.18))
;;   ([mem x threshold]
;;    (let [scores (hd/similarity mem x)
;;          [value index] (into [] (torch/max scores :dim -1))]
;;      (when (<= threshold (py.. value item))
;;        {:idx index
;;         :item (torch/index_select mem -2 index)
;;         :sim value}))))

(comment
  (def mem (hd/seed 2))
  (cleanup-1 mem (py/get-item mem 0))
  (cleanup-idx mem (py/get-item mem 0)))

(defn item-memory->clj
  ([m x threshold] (item-memory->clj m x threshold 1))
  ([{:keys [item->idx idx->item pool]} x threshold n]
   (torch/nonzero (torch/ge (hd/similarity pool x)
                            threshold))))


(defn item-memory->clj*
  [{:keys [item->idx items pool]} x threshold]
  (map (comp items #(py.. % item))
       (torch/nonzero (torch/ge (hd/similarity pool x)
                                threshold))))

(def item-memory->clj (comp first item-memory->clj*))

(defn item-memory-next
  [{:keys [items pool]}]
  (if (< (py.. pool (size 0)) (count items))
    (throw (ex-info "Item memory is full"))
    [(count items) (py/get-item pool (count items))]))

(defn item-memory-clj->vsa
  [mem item]
  (or (when-let [x ((:item->vsa mem) item)] [mem x])
      (let [[idx x] (item-memory-next mem)]
        [(-> mem
             (update :item->vsa assoc item x)
             (update :items conj item)) x])))

(def ^:dynamic *item-memory*
  (atom {:item->vsa {} :items [] :pool (hd/seed 1000)}))

(defn clj->vsa
  [item]
  (let [[new-mem x] (item-memory-clj->vsa @*item-memory*
                                          item)]
    (reset! *item-memory* new-mem)
    x))

(defn vsa->clj
  ([item] (vsa->clj item 0.18))
  ([item threshold]
   (item-memory->clj* @*item-memory* item threshold)))

(def cleanup* vsa->clj)
(def cleanup (comp first cleanup*))

;; -------------------------------------------------------

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

(defn vsa->clj [q])

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

  (cleanup* (clj->vsa* [:. [:* :a :b] :a]))

  (cleanup* (clj->vsa* [:+ :a :b]))

  (cleanup (clj->vsa* [:> :a]))
  (cleanup* (clj->vsa* [:< [:> :a 1]]))


  (cleanup*
   (clj->vsa*
    [:+ [:-- :a 0.5] [:-- :b 0.5]]))

  (cleanup*
   (clj->vsa*
    [:* :c
     [:+
      [:-- :a 0.5]
      [:-- :b 0.5]]]))


  (cleanup* (clj->vsa* [:* :c [:+ :a [:-- :b 0.5]]]))

  (clj->vsa*
   [:?= [:* :c [:+ [:-- :a 0.5] [:-- :b 0.5]]] :c])

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



  (cleanup*
   (clj->vsa*
    [:<
     [:.
      [:*> :a :b :c]
      :a
      [:> :b 1]]
     2]))

  (clj->vsa* [:. [:*> :a :b :c] :a [:> :b 1]])

  (cleanup* (clj->vsa* [:*> :a :b :c]))


  (cleanup*
   (clj->vsa*
    [:. [:*> :a :b :c] :a]))



  (cleanup*
   (clj->vsa*
    [:. [:*> :a :b :c] [:* :a :c]]))



  (cleanup*
   (clj->vsa*
    [:.
     [:*> :a :b :c]
     [:> :b]
     [:>> :c]]))

  (cleanup* (clj->vsa* [:*> :a]))
  (cleanup* (clj->vsa* [:*> :a :b]))

  (hd/similarity
   (hd/permute (clj->vsa* :a))
   (clj->vsa* [:> :a]))

  (hd/similarity
   (hd/permute (clj->vsa* :a))
   (clj->vsa* [:> :a]))



  (clj->vsa* [:*.< :a :b :c])



  (let [a (clj->vsa* :a)
        b (clj->vsa* :b)
        c (clj->vsa* :c)]
    ;; (clj->vsa*
    ;;  [:*.< a b c])
    (let [transition (clj->vsa* [:**> a b c])]
      (clj->vsa* [:*.< transition [:_ b c]])))


  (cleanup* (clj->vsa* [:. [:**> :a :b :c] :b [:> :c]]))

  (cleanup*
   (let [transition (clj->vsa* [:**> :a :b :c])]
     (clj->vsa* [:*.< transition :_ :b :c])))
  '(:a)

  (cleanup*
   (let [transition (clj->vsa* [:**> :a :b :c])]
     (clj->vsa* [:*.< transition :a :_ :c])))
  '(:b)


  (cleanup*
   (let [transition (clj->vsa* [:**> :a [:+ :b1 :b2] :c])]
     (clj->vsa* [:*.< transition :a :_ :c])))
  '(:b1 :b2)




  )
