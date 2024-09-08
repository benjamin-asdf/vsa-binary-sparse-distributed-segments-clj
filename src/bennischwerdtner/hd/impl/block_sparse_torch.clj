(ns bennischwerdtner.hd.impl.block-sparse-torch
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

(defn squeeze-batch
  [tens]
  (if (= 1 (py.. tens (size 0)))
    (torch/squeeze tens 0)
    tens))

;; ----------------------------
;;

(defn indices->hv
  [{:bsdc-seg/keys [segment-count segment-length N]}
   indices]
  (squeeze-batch
    (let [indices (py.. indices (view -1 segment-count))
          batch-size (py.. indices (size 0))]
      (py.. (torch/zeros [batch-size segment-count
                          segment-length]
                         :dtype torch/int8
                         :device *torch-device*)
            (scatter_ -1 (torch/unsqueeze indices -1) 1)
            (view batch-size N)))))

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
(defn hv?
  [{:as opts
    :bsdc-seg/keys [segment-count segment-length N]} x]
  (and (= (type x) :pyobject)
       (= (py/python-type x) :tensor)
       (<= N (py.. x nelement))
       (zero? (fm/mod (py.. x nelement) N))))

;; ----------------

(defn bind-1
  "
  Returns a hdv that is the bind of `a` and `other`.

  `other` is a sequence of tensors.
  "
  [{:as opts
    :bsdc-seg/keys [segment-count segment-length N]} a alpha
   other]

  (let [tens (py.. (torch/stack (vec (concat [a] other)))
               (view -1 segment-count segment-length))
        indices (-> tens
                    (torch/argmax :dim 2))
        _ (py.. (torch/narrow indices
                              0
                              1
                              (dec (py.. indices (size 0))))
            (mul_ alpha))]
    (indices->hv opts
                 (-> indices
                     (torch/sum 0)
                     (torch/remainder segment-length)))))

(defn unbind
  [{:as opts
    :bsdc-seg/keys [segment-count segment-length N]} a
   other]
  (bind-1 opts a -1 other))

(defn bind
  [{:as opts
    :bsdc-seg/keys [segment-count segment-length N]} a
   other]
  (bind-1 opts a 1 other))

;; ----------------------------

(defn permute
  [{:as opts
    :bsdc-seg/keys [segment-count segment-length N]} a n]
  (-> (py.. a (view -1 segment-count segment-length))
      (torch/roll n :dims -1)
      (py.. (view -1 N))
      (squeeze-batch)))

(defn superposition
  [{:bsdc-seg/keys [segment-length segment-count N]} tensor
   tensors]
  (squeeze-batch
   (torch/sum (py.. (torch/stack (vec (concat [tensor] tensors)))
                (view -1 N))
              0)))

(defn thin
  [{:as opts
    :bsdc-seg/keys [segment-count segment-length N]} tens]
  (indices->hv
   opts
   (-> (py.. tens (view -1 segment-count segment-length))
       (torch/argmax :dim 2))))

;; -----------------------------------

(defn similarity
  [{:bsdc-seg/keys [segment-count]} a b]
  (/ (py.. (torch/sum (torch/bitwise_and
                       (py.. a (to :dtype torch/int8))
                       (py.. b (to :dtype torch/int8))))
       item)
     segment-count)
  ;; would be taste,
  ;; After superposition, a vector can have values of more than 1.
  ;; This is useful to model multisets.
  ;; Should this be similar or not similar?
  ;; Probably, it should.
  ;; But user could clamp themselves if they know
  ;;
  ;;
  ;;
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

;; -----------------------------------------------------------

(comment
  (alter-var-root #'*torch-device* (constantly :cpu))
  (alter-var-root #'*torch-device* (constantly :cuda))
  (def default-opts
    "Default opts for (binary sparse distributed, segmented) BSDC-SEG vector operations."
    (let [dimensions (long 1e4)
          segment-count 20]
      {:bsdc-seg/N dimensions
       :bsdc-seg/segment-count segment-count
       :bsdc-seg/segment-length (/ dimensions
                                   segment-count)})))

(comment
  (seed default-opts)
  (time (seed default-opts 500))
  ;; "Elapsed time: 9.136699 msecs"
  ;; "Elapsed time: 9.323494 msecs"
  ;; "Elapsed time: 8.960005 msecs"
  ;; "Elapsed time: 8.967477 msecs"
  (time (torch/sum (bind default-opts
                         (seed default-opts 500))))
  ;; "Elapsed time: 12.376762 msecs"
  ;; "Elapsed time: 12.443287 msecs"
  ;; "Elapsed time: 12.290776 msecs"
  ;; "Elapsed time: 12.58592 msecs"
  )
(comment

  (similarity default-opts
              (seed default-opts)
              (seed default-opts))


  (let [a (seed default-opts)]
    (similarity default-opts a a))
  (let [a (seed default-opts)]
    (similarity default-opts a a))
  (torch/sum (drop-randomly (seed default-opts) 0.5))
  (drop-randomly (seed default-opts) 0.5)
  (time (let [a (seed default-opts)
              a-prime (drop-randomly a 0.5)]
          (dotimes [n 1000]
            (similarity default-opts a a-prime))
          (similarity default-opts a a-prime)))
  ;; "Elapsed time: 189.4513 msecs"
  )

(comment
  (indices->hv
   {:bsdc-seg/N 10
    :bsdc-seg/segment-count 2
    :bsdc-seg/segment-length 5}
   (torch/tensor [0 1] :device *torch-device*))

  (indices->hv
   {:bsdc-seg/N 10
    :bsdc-seg/segment-count 2
    :bsdc-seg/segment-length 5}
   (torch/tensor
    [[0 1]
     [0 2]] :device *torch-device*))

  )

(comment
  (def a (seed default-opts))
  (def b (seed default-opts))
  (def c (bind default-opts a b))
  [
   (bind default-opts a b)
   (similarity default-opts a c)
   (similarity default-opts a a)
   (similarity default-opts a b)
   (similarity default-opts c c)
   (similarity default-opts b c)
   (unbind default-opts c a)
   (unbind default-opts a b)
   (similarity default-opts b
               (unbind default-opts c a))]

  ;; [tensor ([0 0 0 ... 0 0 0] device='cuda:0' dtype=torch.int8)
  ;;  0 1 0 1 0 tensor
  ;;  ([0 0 0 ... 0 0 0] device='cuda:0' dtype=torch.int8) tensor
  ;;  ([0 0 0 ... 0 0 0] device='cuda:0' dtype=torch.int8) 1]


  (bind-1
   {:bsdc-seg/N 10
    :bsdc-seg/segment-count 2
    :bsdc-seg/segment-length 5}
   (torch/tensor
    [0 0 0 1 0
     0 0 0 0 1]
    :device *torch-device*)
   -1
   [
    (torch/tensor
     [0 1 0 0 0
      0 1 0 0 0]
     :device *torch-device*)])

  )

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
  (do (permute {:bsdc-seg/N 10
                :bsdc-seg/segment-count 2
                :bsdc-seg/segment-length 5}
               (torch/tensor [0 0 0 1 0 0 0 0 0 1])
               1))
  (hv? default-opts (seed default-opts))
  (hv? default-opts (seed default-opts 2))
  (hv? default-opts (torch/tensor [0])))
