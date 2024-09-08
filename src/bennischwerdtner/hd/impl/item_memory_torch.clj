(ns bennischwerdtner.hd.impl.item-memory-torch
  (:require [tech.v3.datatype.functional :as f]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
            [tech.v3.datatype.bitmap :as bitmap]
            [fastmath.random :as fm.rand]
            [fastmath.core :as fm]
            [tech.v3.datatype.unary-pred :as unary-pred]
            [tech.v3.datatype.argops :as dtype-argops]
            [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python :refer [py. py..] :as py]
            [bennischwerdtner.hd.prot :refer [ItemMemory m-clj->vsa m-cleanup m-cleanup* m-cleanup-verbose] :as prot]
            [bennischwerdtner.pyutils :as pyutils :refer
             [*torch-device*]]))

(try
  (require-python '[numpy :as np])
  (require-python '[torch :as torch])
  (require-python '[torch.sparse :as torch.sparse])
  (require '[libpython-clj2.python.np-array])
  true
  (catch Exception e false))

(defn ->codebook-matrix
  [pool]
  (let [pool (if (vector? pool) (torch/stack pool) pool)]
    (py.. pool (to :dtype torch/float16))))

;; This is the overlap, is relative to each other interesting
(defn codebook-weights
  [codebook-matrix q]
  (torch/mv codebook-matrix
            (py.. (pyutils/ensure-torch q)
              (to :dtype torch/float16))))

(defn codebook-max
  [codebook-matrix q]
  (py.. (torch/argmax (codebook-weights codebook-matrix q)) (item)))

(defn codebook-cleanup-verbose
  [{:bsdc-seg/keys [segment-count]} codebook-matrix q
   threshold]
  (let [sims (torch/div (codebook-weights codebook-matrix q)
                        segment-count)
        mask (torch/ge sims threshold)
        idxs (py.. (torch/nonzero mask) (squeeze 1))]
    {:idxs idxs
     :sims (py.. (py/get-item sims mask)
             (to :dtype torch/float32))}))

;; ---------------------------------
