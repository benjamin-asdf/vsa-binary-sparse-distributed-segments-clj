(ns bennischwerdtner.hd.codebook-item-memory
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
   [tech.v3.datatype.argops :as dtype-argops]
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python :refer [py. py..] :as py]
   [bennischwerdtner.hd.prot :refer
    [ItemMemory m-clj->vsa m-cleanup m-cleanup*
     m-cleanup-verbose] :as prot]
   [bennischwerdtner.pyutils :as pyutils :refer
    [*torch-device*]]
   [bennischwerdtner.hd.hdc :as hdc]))


(def available
  (try (require-python '[numpy :as np])
       (require-python '[torch :as torch])
       (require-python '[torch.sparse :as torch.sparse])
       true
       (catch Exception e false)))

;; ah, whatever.
;; (I wanted to check for 'available')
;;

(defn ->codebook-matrix
  [pool]
  (py.. (pyutils/ensure-torch pool)
    (to :dtype torch/float16)))

(defn codebook-cleanup-verbose
  [codebook-matrix q threshold]
  (let [sims (torch/div
               (torch/mv codebook-matrix
                         (py.. (pyutils/ensure-torch q)
                               (to :dtype torch/float16)))
               (:bsdc-seg/segment-count hd/default-opts))
        mask (torch/ge sims threshold)
        idxs (py.. (torch/nonzero mask) (squeeze 1))]
    {:idxs (pyutils/torch->jvm idxs)
     :sims (pyutils/torch->jvm (py.. (py/get-item sims mask)
                                     (to :dtype
                                         torch/float32)))}))

#_(defn codebook-cleanup*
  [codebook-matrix q threshold]
  (let [sims (torch/div
               (torch/mv codebook-matrix
                         (py.. (pyutils/ensure-torch q)
                               (to :dtype torch/float16)))
               (:bsdc-seg/segment-count hd/default-opts))
        mask (torch/ge sims threshold)
        idxs (py.. (torch/nonzero mask) (squeeze 1))]
    (pyutils/torch->jvm idxs)))


(defn codebook-item-memory
  [n-seeds]
  (let [lut (atom {})
        pool (hdc/preallocated-alphabet n-seeds)
        seed (fn []
               (if (<= n-seeds (count @lut))
                 (throw (Exception.
                          "codebook out of seeds"))
                 (nth pool (count @lut))))
        codebook-matrix (->codebook-matrix pool)]
    (reify
      prot/ItemMemory
        (m-clj->vsa [this item]
          (or (get @lut item)
              (let [v (seed)]
                (swap! lut assoc item v v item)
                v)))
        (m-cleanup-verbose [this q]
          (m-cleanup-verbose this q 0.18))
        (m-cleanup-verbose [this q threshold]
          (let [{:keys [idxs sims]}
                  (codebook-cleanup-verbose codebook-matrix
                                            q
                                            threshold)]
            (map (fn [hd sim]
                   {:k (get @lut hd) :similarity sim :v hd})
              (dtt/select pool idxs)
              sims)))
        (m-cleanup [this q]
          (get @lut
               (nth pool
                    (py. (torch/argmax
                           (torch/mv
                             codebook-matrix
                             (py.. (pyutils/ensure-torch q)
                                   (to :dtype
                                       torch/float16))))
                         item))))
        (m-cleanup* [this q threshold]
          (map :k (m-cleanup-verbose this q threshold)))
        (m-cleanup* [this q] (m-cleanup* this q 0.18)))))


(comment
  (def m (codebook-item-memory 10))
  (prot/m-cleanup-verbose m (prot/m-clj->vsa m :foo) 0.2))


(comment (torch/argmax
          (torch/mv
           codebook-matrix
           (py.. (pyutils/ensure-torch q)
             (to :dtype torch/float16))))

         (torch/argsort
          (torch/mv
           (torch/tensor
            [[0 1 0]
             [0 1 1]])
           (torch/tensor [0 1 1])))

         (py..
             (torch/argmax (torch/mv (torch/tensor [[0 1 0]
                                                    [0 1 1]])
                                     (torch/tensor [0 1 1])))
             item)

         (torch/div
          (torch/mv (torch/tensor [[0 1 0 0] [0 1 1 1]])
                    (torch/tensor [0 1 1 0]))
          2))
