(ns bennischwerdtner.hd.item-memory
  (:require [tech.v3.datatype.functional :as f]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
            [tech.v3.datatype.bitmap :as bitmap]
            [fastmath.random :as fm.rand]
            [fastmath.core :as fm]
            [bennischwerdtner.pyutils :as pyutils :refer
             [*torch-device*]]
            [bennischwerdtner.hd.core :as hd]
            [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python :refer [py. py..] :as py]
            [tech.v3.datatype.unary-pred :as unary-pred]
            [tech.v3.datatype.argops :as dtype-argops]
            [bennischwerdtner.hd.impl.item-memory-torch :as
             item-memory-torch]))

;; --------------------------

#_(defprotocol CleanupMemory
  (-cleanup-1 [this query]
              [this query threshold])
  (-cleanup-verbose [this query threshold]))

(defprotocol Codebook
  (-cleanup [this query]
            [this query threshold])
  (-cleanup-verbose [this query threshold]))

(defprotocol ItemMemory
  (m-clj->vsa [this item])
  (m-cleanup-verbose [this q]
                     [this q threshold])
  (m-cleanup
    [this q]
    [this q threshold])
  (m-cleanup* [this q]
    [this q threshold]))

;; ----------------------------------------

(defn cleanup-verbose-impl
  [codebook-matrix q threshold idx->item]
  (let [{:keys [idxs sims]}
          (item-memory-torch/codebook-cleanup-verbose
            hd/default-opts
            codebook-matrix
            q
            threshold)]
    (sort-by :similarity
             (fn [a b] (compare b a))
             (into []
                   (map (fn [idx sim]
                          {:i idx
                           :k (get idx->item idx)
                           :similarity sim})
                     (pyutils/ensure-jvm idxs)
                     (pyutils/ensure-jvm sims))))))

;; -----------------------------------------

(defn codebook-item-memory-1
  [pool]
  (let [item->idx (atom {})
        idx->item (atom [])
        seed
          (fn []
            (if (<= (py.. pool (size 0)) (count @item->idx))
              (throw (Exception. "codebook out of seeds"))
              (py/get-item pool (count @item->idx))))
        codebook-matrix (item-memory-torch/->codebook-matrix
                          pool)]
    (reify
      ItemMemory
      (m-clj->vsa [this item]
        (if-let [hd-idx (get @item->idx item)]
          (py/get-item pool hd-idx)
          (let [hdv (seed)]
            (swap! item->idx assoc
                   item
                   (count @item->idx))
            (swap! idx->item conj item)
            hdv)))
      (m-cleanup-verbose [this q]
        (m-cleanup-verbose this q 0.18))
      (m-cleanup-verbose [this q threshold]
        (cleanup-verbose-impl codebook-matrix
                              q
                              threshold
                              @idx->item))
      ;; not that this doesn't take a threshold into account
      (m-cleanup [this q]
        (let [idx (item-memory-torch/codebook-max
                   codebook-matrix
                   q)]
          (get @idx->item idx)))
      (m-cleanup [this q threshold]
        (first (m-cleanup* this q threshold)))
      (m-cleanup* [this q threshold]
        (map :k
             (cleanup-verbose-impl codebook-matrix
                                   q
                                   threshold
                                   @idx->item)))
      (m-cleanup* [this q] (m-cleanup* this q 0.18)))))

(defn codebook-item-memory
  [n-seeds]
  (if (<= n-seeds 1)
    (throw (Exception. "codebook-item-memory n-seeds must be > 1")))
  (codebook-item-memory-1 (hd/seed n-seeds)))
