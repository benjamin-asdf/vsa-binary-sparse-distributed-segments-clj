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

(defn indices->hv
  [{:bsdc-seg/keys [segment-count segment-length N]}
   indices]
  (let [indices (py.. indices (view -1 segment-count))
        batch-size (py.. indices (size 0))]
    (py.. (torch/zeros [batch-size segment-count
                        segment-length]
                       :dtype torch/int8
                       ;; :dtype torch/float32
                       :device *torch-device*)
          (scatter_ -1 (torch/unsqueeze indices -1) 1)
          (view batch-size N))))

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

(defn ->empty
  ([opts] (->empty opts 1))
  ([{:as opts
     :bsdc-seg/keys [segment-count segment-length N]}
    batch-size]
   (torch/zeros [batch-size N] :device *torch-device*)))

(defn hv?
  [{:as opts
    :bsdc-seg/keys [segment-count segment-length N]} x]
  (and (= (type x) :pyobject)
       (= (py/python-type x) :tensor)
       (<= N (py.. x nelement))
       (zero? (fm/mod (py.. x nelement) N))))

;; ----------------

(defn reshape-to-N
  [x {:as opts :bsdc-seg/keys [N]}]
  (if (hv? opts x)
    [(py.. x (view -1 N))]
    (mapcat #(reshape-to-N % opts) x)))

(defn bind-1
  "
  Returns a hdv that is the bind of `a` and `other`.

  `other` is a sequence of tensors.
  "
  [{:as opts
    :bsdc-seg/keys [segment-count segment-length N]} a alpha
   other]
  (let [tens (py.. (torch/stack
                     (vec (concat (reshape-to-N a opts)
                                  (reshape-to-N other
                                                opts))))
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
      (py.. (view -1 N))))

(defn superposition
  [{:as opts
    :bsdc-seg/keys [segment-length segment-count N]} a
   other]
  (torch/sum (py.. (torch/stack
                     (vec (concat (reshape-to-N a opts)
                                  (reshape-to-N other
                                                opts))))
                   (view -1 segment-count segment-length))
             0))

(defn thin
  [{:as opts
    :bsdc-seg/keys [segment-count segment-length N]} tens]
  (indices->hv
    opts
    (-> (py.. tens (view -1 segment-count segment-length))
        (torch/argmax :dim 2))))

;; -----------------------------------

(defn dot-similarity
  [a others]
  (let [a (py.. a (to :dtype torch/int8))
        others (py.. others (to :dtype torch/int8))]
    (torch/sum (torch/bitwise_and a others) :dim -1)))

(defn similarity
  [{:bsdc-seg/keys [segment-count]} a others]
  (torch/div (dot-similarity a others) segment-count))

;; -------------------------------------

(defn drop-randomly
  [a drop-prob]
  (torch/where (torch/lt (torch/rand (py.. a (size))
                                     :device
                                     *torch-device*)
                         drop-prob)
               (torch/zeros_like a)
               a))

(defn keep-segments
  [{:bsdc-seg/keys [segment-count segment-length N]} a
   segment-keep-count]
  (let [segment-indices (torch/arange segment-count
                                      :device
                                      *torch-device*)
        kept-segments (torch/ge segment-indices
                                segment-keep-count)
        mask (torch/repeat_interleave kept-segments
                                      segment-length)
        mask (torch/broadcast_to mask (py.. a (size)))]
    (torch/where mask a (torch/zeros_like a))))

;; -----------------------------------------------------------

;; This version stops working when num-levels > segment-count.
(defn level
  [{:as opts :bsdc-seg/keys [segment-count]} num-levels]
  (let [seeds (seed opts 2)
        bin-len (/ segment-count num-levels)]
    (torch/stack
     (into []
           (for [n (range num-levels)]
             (let [segments-a (* bin-len n)
                   segments-b (- segment-count segments-a)]
               (superposition
                opts
                (keep-segments opts
                               (py/get-item seeds 0)
                               segments-a)
                [(keep-segments opts
                                (py/get-item seeds 1)
                                segments-b)])))))))

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
                       0 2 0 0 1]
                      :device *torch-device*))
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
