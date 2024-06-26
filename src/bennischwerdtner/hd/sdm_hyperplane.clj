(ns bennischwerdtner.hd.sdm-hyperplane
  (:require
   [clojure.test :as t]
   [tech.v3.datatype.functional :as f]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [tech.v3.datatype.bitmap :as bitmap]
   [fastmath.random :as fm.rand]
   [fastmath.core :as fm]
   [tech.v3.datatype.unary-pred :as unary-pred]
   [tech.v3.datatype.argops :as dtype-argops]))


;; Jaeckel, L.A. 1989a. An Alternative Design for a Sparse Distributed Memory.
;; Report RIACS TR 89.28, Research Institute for Advanced Computer Science,
;; NASA Ames Research Center.
;;
;; Jaeckel, L.A. 1989b. A Class of Designs for a Sparse Distributed Memory. Report
;; RIACS TR 89.30, Research Institute for Advanced Computer Science, NASA
;; Ames Research Center.


(def counter-range [-15 15])

;; -------------------
;; Address Matrix A
;; -------------------

;; in hyperplane design, each row has 3 (k = 3) ones and (N - k = 997) zeros

;; In Cerebellum, this is ~5 per granule cell,
;; motivating the idea to make this sparse
;;
(defn ->address-matrix
  [word-length hard-address-location-count
   address-density-k]
  (let [ones (into []
                   (repeatedly
                    hard-address-location-count
                    (fn []
                      (into #{}
                            (take address-density-k
                                  (shuffle
                                   (range
                                    word-length)))))))]
    (dtype/clone (dtt/compute-tensor
                  [hard-address-location-count word-length]
                  (fn [i j] (if ((ones i) j) 1 0))
                  :int8))))



;; -------------------------------
;; Address Decoder
;; -------------------------------
;;
;; Input:
;; - An address of N bits,
;; - The address matrix A
;;
;; Output: x, the address location vector of size [M],
;; with each 'on' address location being set 1
;;
;; x might also be represented as set of indices.
;;

;;
;; In hyperplane, all the bits for one decoder cell
;; have to be 'on', then it is active
;; I just count to address-density-k

(defn decode-addresses
  [address-matrix address address-density-k]
  ;; d
  (let [inputs
        (dtt/reduce-axis
         address-matrix
         (fn [row]
           (f/dot-product row address)))
        activations (f/<=
                     address-density-k
                     inputs)]
    ;; (dtt/->tensor activations {:datatype :int8})
    (unary-pred/bool-reader->indexes activations)))



;; --------------------------
;; Storage
;; --------------------------
;;
;; The content matrix:
;;
;; M x U counters
;; c - the range counter bits, e.g. -15...15
;;     if c={0,1}, then only the last write is remembered
;;
;; c is -15...15 here

(defn ->content-matrix
  [word-length hard-address-location-count]
  (dtt/reshape (dtt/->tensor (dtype/alloc-zeros
                              :int8
                              (*
                               hard-address-location-count
                               word-length)))
               [hard-address-location-count word-length]))


;; returns the updated content matrix
;; for each location in address-locations,
;; update the bit counter using 'input-word'
;; incrementing for 1 and decrementing for 0 in input word
;; but stay within counter-range
(defn write!
  [content-matrix address-locations input-word]
  (let [C content-matrix
        input-update (f/- (f/* input-word 2) 1)
        updated-rows (dtt/map-axis
                       (dtt/select C address-locations)
                       (fn [row]
                         (-> (f/+ row input-update)
                             (f/max (first counter-range))
                             (f/min (second
                                      counter-range))))
                       1)]
    (doseq [l address-locations
            row updated-rows]
      (dtt/mset! C l row))
    C))


;; 1. sum up the address location bits of activated rows
;; 2. create the output word: positive -> 1, negative -> 0
;;
;; Each of the bits in the output models 1 Purkinje cell (output cell)
;;
(defn read
  [content-matrix address-locations]
  (dtt/->tensor
   ;; ties? probably doesn't matter because it is so
   ;; unlikely
   (f/< 0
        (dtt/reduce-axis (dtt/select content-matrix
                                     address-locations)
                         f/sum
                         0))
   {:datatype :int8}))

;; addresses are skewed,
;; e.g. L = 100
;; N = 1000
(defn ->word
  [N L]
  (dtt/clone
   (dtt/compute-tensor
    [N]
    (comp
     {false 0 true 1}
     boolean
     (into #{} (take L (shuffle (range N)))))
    :int8)))
