(ns bennischwerdtner.hd.block-sparse-torch
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
    [*torch-device*]]))

(do
  (require-python '[numpy :as np])
  (require-python '[torch :as torch])
  (require-python '[torch.sparse :as torch.sparse])
  (require '[libpython-clj2.python.np-array]))

;; ----------------------------
;;

(defn squeeze-batch
  [tens]
  (if (= 1 (py.. tens (size 0)))
    (torch/squeeze tens 0)
    tens))

(defn indices->hv
  [{:bsdc-seg/keys [segment-count segment-length N]}
   indices]
  (squeeze-batch
    (let [indices (py.. indices (view -1 segment-count))
          batch-size (py.. indices (size 0))]
      (py.. (torch/zeros
             [batch-size segment-count
              segment-length]
             :dtype torch/int8
             :device *torch-device*)
            (scatter_ -1 (torch/unsqueeze indices -1) 1)
            (view batch-size N)))))

(comment
  (indices->hv
   {:bsdc-seg/N 10
    :bsdc-seg/segment-count 2
    :bsdc-seg/segment-length 5}
   (torch/tensor
    [0 1]))

  (indices->hv
   {:bsdc-seg/N 10
    :bsdc-seg/segment-count 2
    :bsdc-seg/segment-length 5}
   (torch/tensor
    [0 2]))

  (indices->hv
   {:bsdc-seg/N 10
    :bsdc-seg/segment-count 2
    :bsdc-seg/segment-length 5}
   (torch/tensor
    [[0 1]
     [0 2]])))

(defn seed
  ([opts] (seed opts 1))
  ([{:as opts
     :bsdc-seg/keys [segment-count segment-length N]}
    batch-size]
   (indices->hv opts
                (torch/tensor
                 (into []
                       (for [_ (range batch-size)]
                         (vec (repeatedly
                               segment-count
                               #(fm.rand/irand
                                 segment-length)))))
                 :device *torch-device*))))


(comment
  (def segment-length 5)
  (def segment-count 2)
  (def N 10)
  (time (seed default-opts)))

;; ----------------

(defn bind
  [{:as opts
    :bsdc-seg/keys [segment-count segment-length N]} a &
   other]
  (indices->hv opts
               (-> (torch/stack (vec (concat [a] other)))
                   (py..
                     (view -1 segment-count segment-length))
                   (torch/argmax :dim 2)
                   (torch/sum 0)
                   (torch/fmod segment-length))))

;; ----------------------------

(defn permute
  [{:as opts
    :bsdc-seg/keys [segment-count segment-length N]} a n]
  (-> (py.. a (view -1 segment-count segment-length))
      (torch/roll n :dims -1)
      (py.. (view -1 N))
      (squeeze-batch)))

(comment
  (do
    (permute
     {:bsdc-seg/N 10
      :bsdc-seg/segment-count 2
      :bsdc-seg/segment-length 5}
     (torch/tensor
      [0 0 0 1 0
       0 0 0 0 1])
     1)))

(defn superposition [& tensors]
  (torch/sum (torch/stack (vec tensors)) 0))

(defn thin
  [{:as opts
    :bsdc-seg/keys [segment-count segment-length N]} tens]
  (indices->hv
    opts
    (-> (py.. tens (view -1 segment-count segment-length))
        (torch/argmax :dim 2))))

(comment
  (thin {:bsdc-seg/N 10
         :bsdc-seg/segment-count 2
         :bsdc-seg/segment-length 5}
        (torch/tensor [0 2 0 1 0
                       ;;
                       0 2 0 0 1]))
  ;; ------------------------
  (thin {:bsdc-seg/N 10
         :bsdc-seg/segment-count 2
         :bsdc-seg/segment-length 5}
        (torch/tensor [[0 2 0 1 0 0 2 0 0 1]
                       [0 2 0 1 0 0 2 0 0 1]])))

(comment
  (torch/roll
   (torch/tensor
    [[[1 2 3]
      [1 2 3]]
     [[4 5 6]
      [1 2 3]]])
   (- 1)
   :dims -1)
  (time
   (bind
    default-opts
    (seed default-opts)
    (seed default-opts)
    (seed default-opts)
    (seed default-opts)))
  )

;; -----------------------------------

(defn similarity
  [{:bsdc-seg/keys [segment-count]} a b]
  (/
   (py.. (torch/sum (torch/bitwise_and a b)) item)
   segment-count)
  ;; would be taste
  ;; (/ (py.. (torch/sum (torch/bitwise_and
  ;;                       (torch/clamp_max a 1)
  ;;                       (torch/clamp_max b 1)))
  ;;          (item))
  ;;    segment-count)
  )

;; -------------------------------------

(defn drop-randomly
  [a drop-prob]
  (torch/where (torch/lt (torch/rand (py.. a (size))
                                     :device
                                     *torch-device*)
                         drop-prob)
               (torch/zeros_like a)
               a))



















(comment
  (do (defn bind
        [{:as opts
          :bsdc-seg/keys [segment-count segment-length N]} a &
         other]
        (indices->hv
         opts
         (-> (torch/stack (vec (concat [a] other)))
             (py.. (view -1 segment-count segment-length))
             (torch/argmax :dim 2)
             (torch/sum 0)
             (torch/fmod segment-length))))
      (def default-opts
        {:bsdc-seg/N 10
         :bsdc-seg/segment-count 2
         :bsdc-seg/segment-length 5})
      ;; (bind
      ;;  default-opts
      ;;  (seed default-opts)
      ;;  (seed default-opts)
      ;;  (seed default-opts))
      (bind default-opts
            (torch/tensor
             [0 0 0 1 0
              ;;
              0 0 0 0 1]
             :device
             *torch-device*)
            (torch/tensor [0 1 0 0 0
                           ;;
                           0 1 0 0 0]
                          :device
                          *torch-device*))
      ;; tensor([0., 0., 0., 0., 1., 1., 0., 0., 0., 0.], device='cuda:0')
      ;;
      )
  (-> (torch/stack [(torch/tensor [0 0 0 0 1 0 0 0 0 1]
                                  :device
                                  *torch-device*)
                    (torch/tensor [0 0 0 0 1 0 0 0 0 1]
                                  :device
                                  *torch-device*)])
      (py.. (view -1 2 5))
      (torch/argmax :dim 2)
      (torch/sum :dim 1)
      (torch/fmod 5)))









(comment

  (def default-opts
    "Default opts for (binary sparse distributed, segmented) BSDC-SEG vector operations."
    (let [dimensions (long 1e4)
          segment-count 20]
      {:bsdc-seg/N dimensions
       :bsdc-seg/segment-count segment-count
       :bsdc-seg/segment-length (/ dimensions
                                   segment-count)}))



  (time (seed default-opts 500))
  ;; "Elapsed time: 9.136699 msecs"
  ;; "Elapsed time: 9.323494 msecs"
  ;; "Elapsed time: 8.960005 msecs"
  ;; "Elapsed time: 8.967477 msecs"


  (time (torch/sum
         (bind
          default-opts
          (seed default-opts 500))))
  ;; "Elapsed time: 12.376762 msecs"
  ;; "Elapsed time: 12.443287 msecs"
  ;; "Elapsed time: 12.290776 msecs"
  ;; "Elapsed time: 12.58592 msecs"








  (let
      [idx (torch/argmax (torch/rand [2 5]) 1)]
      [idx
       (torch/add idx (torch/mul 5 (torch/arange 2)))
       (torch/index_fill
        (torch/zeros [10])
        0
        (torch/add idx (torch/mul 5 (torch/arange 2)))
        1)])


  (defn indices->hv
    [{:bsdc-seg/keys [segment-count segment-length N]}
     indices]
    (let [batch-size (py.. indices (size 0))
          device *torch-device*
          result (torch/zeros [batch-size segment-count segment-length]
                              :device device)
          batch-indices (torch/arange batch-size :device device)
          segment-indices (-> (torch/arange segment-count :device device)
                              (torch/unsqueeze 0)
                              (torch/repeat_interleave batch-indices 1))
          expanded-indices (torch/unsqueeze indices 2)]
      (-> (py/set-item!
           result
           [batch-indices segment-indices expanded-indices]
           1)
          (torch/reshape [batch-size N]))))


  (torch/repeat_interleave (torch/) )

  (let
      [batch-size 2
       segment-count 2
       segment-length 5
       indices (torch/tensor [[1 2] [3 4]])]
      (py/set-item!
       (torch/zeros [batch-size segment-count segment-length]
                    :device
                    *torch-device*)
       [
        (torch/arange segment-count :device *torch-device*)
        indices]
       1)))

(comment
  (let [segment-count 2
        segment-length 5
        N 10
        indices (torch/tensor [0 1])
        indices (py.. indices (view -1 segment-count))
        batch-size (py.. indices (size 0))]
    ;; (torch/unsqueeze indices -1)
    (py.. (torch/zeros [batch-size segment-count
                        segment-length])
      (scatter_ -1 (torch/unsqueeze indices -1) 1)
      (view batch-size N))))

(comment

  (torch/ge
   (torch/add
    (torch/tensor [1 0 1 0 0 0 1 0 1 0] :device *torch-device*)
    (torch/tensor [1 0 1 0 1 0 0 0 1 0] :device *torch-device*))
   (torch/tensor 2))



  (similarity
   default-opts
   (seed default-opts)
   (seed default-opts))

  (let [a (seed default-opts)]
    (similarity default-opts a a))

  (let [a (seed default-opts)
        ]
    (similarity default-opts a a))
  (torch/sum (drop-randomly (seed default-opts) 0.5))


  (drop-randomly (seed default-opts) 0.5)


  (time
   (let [a (seed default-opts)
         a-prime (drop-randomly a 0.5)]
     (dotimes [n 1000]
       (similarity default-opts a a-prime))
     (similarity default-opts a a-prime)))

  ;; "Elapsed time: 189.4513 msecs"



  )
