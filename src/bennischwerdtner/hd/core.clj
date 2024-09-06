(ns bennischwerdtner.hd.core
  (:refer-clojure :exclude [drop])
  (:require [tech.v3.datatype.unary-pred :as unary-pred]
            [tech.v3.datatype.functional :as f]
            [fastmath.random :as fm.rand]
            [fastmath.core :as fm]
            [tech.v3.datatype.argops :as dtype-argops]))


;; --------------------

(def default-opts
  "Default opts for (binary sparse distributed, segmented) BSDC-SEG vector operations."
  #_(let [dimensions (long 1e4)
          ;; Rachkovskij (2001) showed that this value
          ;; works well, therefore we use it
          density-probability (/ 1 (fm/sqrt dimensions))
          ;; segment count == non-zero bits count
          segment-count (long (* dimensions
                                 density-probability))]
      {:bsdc-seg/N dimensions
       :bsdc-seg/density-probability density-probability
       :bsdc-seg/segment-count segment-count
       :bsdc-seg/segment-length (/ dimensions
                                   segment-count)})
  ;; I started enjoying the daringness of making this
  ;; 20
  (let [dimensions (long 1e4)
        segment-count 20]
    {:bsdc-seg/N dimensions
     :bsdc-seg/segment-count segment-count
     :bsdc-seg/segment-length (/ dimensions
                                 segment-count)}))


;; --------------------

(defprotocol HDV
  (-bind [this & other])
  (-permute [this n])
  (-superposition [this & other])
  (-drop [this drop-chance])
  (-hv? [this])
  (-similarity [this other]))

;; ----------------------
