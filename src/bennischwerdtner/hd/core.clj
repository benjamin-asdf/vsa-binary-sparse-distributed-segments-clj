(ns bennischwerdtner.hd.core
  (:refer-clojure :exclude [drop])
  (:require
   [tech.v3.datatype.unary-pred :as unary-pred]
   [tech.v3.datatype.functional :as f]
   [fastmath.random :as fm.rand]
   [fastmath.core :as fm]
   [tech.v3.datatype.argops :as dtype-argops]
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python :refer [py. py..] :as py]
   [bennischwerdtner.hd.impl.item-memory-torch :as
    item-memory-torch]
   [bennischwerdtner.hd.impl.block-sparse-torch :as
    impl.torch]))

;; --------------------

(def default-opts
  "Default opts for (binary sparse distributed, segmented) BSDC-SEG vector operations."
  ;; (let [dimensions (long 1e4)
  ;;         ;; Rachkovskij (2001) showed that this
  ;;         value ;; works well, therefore we use it
  ;;         density-probability (/ 1 (fm/sqrt
  ;;         dimensions))
  ;;         ;; segment count == non-zero bits count
  ;;         segment-count (long (* dimensions
  ;;                                density-probability))]
  ;;     {:bsdc-seg/N dimensions
  ;;      :bsdc-seg/density-probability
  ;;      density-probability :bsdc-seg/segment-count
  ;;      segment-count :bsdc-seg/segment-length (/
  ;;      dimensions
  ;;                                  segment-count)})
  ;; I started enjoying the daringness of making this
  ;; 20
  (let [dimensions (long 1e4)
        segment-count 20]
    {:bsdc-seg/N dimensions
     :bsdc-seg/segment-count segment-count
     :bsdc-seg/segment-length (/ dimensions
                                 segment-count)}))

(def implementations #{:torch})

(def default-implementation :torch)

;; --------------------

(defprotocol HDV
  (-bind [x other])
  (-unbind [x other])
  (-permute [x n])
  (-superposition [x other])
  (-drop [x drop-chance])
  (-hv? [x])
  (-similarity [x other])
  (-thin [x]))

;; ----------------------
;; torch
;; This better by a python torch hdv or whuumps!

(extend-protocol HDV
  Object
    (-hv? [x] (impl.torch/hv? default-opts x))
    (-permute [x n] (impl.torch/permute default-opts x n))
    (-superposition [x other]
      (impl.torch/superposition default-opts x other))
    (-drop [x drop-chance]
      (impl.torch/drop-randomly x drop-chance))
    (-similarity [x y]
      (impl.torch/similarity default-opts x y))
    (-bind [x other]
      (impl.torch/bind default-opts x other))
    (-unbind [x other]
      (impl.torch/unbind default-opts x other))
    (-thin [x] (impl.torch/thin default-opts x)))

;; ------------------------------

(defn seed
  ([] (seed 1))
  ([batch-size]
   (case default-implementation
     :torch (impl.torch/seed default-opts batch-size))))

(defn bind
  ([x] (-bind x []))
  ([x other] (-bind x [other]))
  ([x y & other] (-bind x (concat [y] other))))

(defn unbind
  ([x] (-unbind x []))
  ([x other] (-unbind x [other]))
  ([x y & other] (-unbind x (concat [y] other))))

(defn permute
  ([x] (permute x 1))
  ([x n] (-permute x n)))

(defn permute-invserse ([x] (-permute x -1)))

(defn superposition
  ([x] (-superposition x []))
  ([tensor & tensors] (-superposition tensor tensors)))

(defn drop [x drop-chance]
  (-drop x drop-chance))

(defn hv? [x]
  (-hv? x))

(defn similarity [x other]
  (-similarity x other))

(defn thin [x] (-thin x))

;; -----------------------------

;; (defprotocol Codebook
;;   (cleanup
;;     [this query]
;;     [this query threshold])
;;   ())



(comment
  (py.. (first (seed 10)) (nelement))
  (similarity (seed) (seed))
  (let [a (impl.torch/seed default-opts)]
    (-similarity (-permute a 1) (-permute a 1)))
  (let [a (impl.torch/seed default-opts)]
    (-similarity (-permute (-permute a 1) -1) a))

  (bind (seed) (seed)))


;; ----------------------
