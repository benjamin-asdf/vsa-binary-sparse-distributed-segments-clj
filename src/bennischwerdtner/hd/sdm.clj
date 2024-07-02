(ns bennischwerdtner.hd.sdm
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


;; ---
;; Addres Register
;; Word-In Register
;; Word-out Register
;; Address Matrix A
;;  M addresses
;;  this is implicit in hardware,
;;  it comes from the address decoder circuit
;;
;; Contents Matrix
;;  M x U bits
;;  is an array of one-bit registers or flip-flops.
;;
;;


;; RAM
;;
;; I
;; The Location:
;; address -> contents

;; addr decoder:
;; address in address register e.g. 20 bits
;;
;; 2^20 locations:
;;

;; (defn decode-address [address] ) ; returns location
;; y - [0 0 0 1 0 0, ...]
;; y is a location mask called activation vector
;; active location

;; 2^20 locations means 20 bits address
;; N = 20
;; capacity of a location is called word size
;; capactiy of the memory is the word size multiplied by the
;; number of memory locations M x U bits
;;

;; storage and retrieval on word at a time in
;; address register (N bit memory array)
;; word-in register: U-bit word to store
;; word-out register: U-bit word to retrieve


;; SDM
;; same thing
;; but the registers are large N = 1k
;;


;;
;; it's impossible to to make 2^N address hard locations
;;

;; if M is the size of the sample,
;; we want a memory with M locations

;;
;; for each address, you chose a subset of hard locations
;; different ways of chosing them can be considered
;;

;; address matrix:
;; make 0 and 1 same likely
;; this is not how granular cells work in cerebellum, they are sparse
;; and in fact, a sparse A is even better
;;



;; disagreement:
;; granule cells have 3-7 inputs, not N like here
;; -> A is sparse
;; 1 - 10% input lines activate at once (1% maybe high)
;; in the model that would be 50%, if activation is a 1
;;
;; -> overcome mathematically by Jaeckles hyperplane design
;; address is sparse, very small number active. (~ mossy fibres)
;;
;; A random addr. matrix (mossy fibre -> granule cell synapses)
;;
;; and this is mathematically the better way to do it!


;; C is also very sparse,
;; granule cells -> purkinje cells is 100k out of billions

;; N - word length
;; M - count hard address locations (memory size) e.g. 1.000.000

;; T - Training size e.g. 10.000
;;
;; p - probability of activation (depends on implementation)
;; "ideally" 2MT^-1/3
;; e.g. 0.000445

;; H - radius of activation e.g. Hamming 447

(comment
  ;; I don't really get how they get 0.000445
  (Math/pow (* 2 1e6 1e4) (/ -1 3))
  3.684031498640388E-4)


(def activation-radius-hamming 447)
(def counter-range [-15 15])

;;
;; address-density is 0.5 here, this is sparse in hyperplane design (Jaeckle),
;; where k = 10, N = 1000, p = 0.5^k, p = 0.001
;;

(defn ->address-matrix
  [word-length hard-address-location-count address-density]
  (dtype/clone
   (dtt/compute-tensor
    [hard-address-location-count word-length]
    (fn [i j] (fm.rand/flip address-density))
    :int8)))

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


(defn hamming-dist [x y]
  (f/sum (f/bit-xor x y)))


(defn decode-addresses
  [address-matrix address activation-radius-hamming]
  ;; d
  (let [hamming-distances (dtt/reduce-axis
                           address-matrix
                           #(hamming-dist % address))
        ;; threshold the hamming dists -> y
        activations (f/<= hamming-distances
                          activation-radius-hamming)]
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

;; 50% 0 and 1
(defn ->word
  [N]
  (dtt/clone
   (dtt/compute-tensor [N] (fn [_] (fm.rand/flip)) :int8)))

(comment

  (def A (->address-matrix 100 100 0.5))
  (dtt/reduce-axis (dtt/->tensor [[0 0 1] [0 1 1]]) f/sum)
  (f/<= 1 [0 1])

  (dtt/reduce-axis
   [[1 0 1 0] [0 1 1 0]]
   #(hamming-dist % [0 0 0 0]))

  (decode-addresses
   ;; A
   [[1 0 1 0]
    [1 0 1 0]
    [1 0 1 0]
    [0 1 1 0]]
   ;; x (address)
   [0 1 0 1]
   ;; dist
   3)

  (write
   ;; C
   [[0 0]
    [0 0]]
   ;; addr locations
   [1 1]
   ;; input word
   [0 1]
   )


  (let [C (dtt/->tensor [[0 1 0] [0 1 0] [0 3 2]])
        input-word [0 0 1]
        input-update (f/- (f/* input-word 2) 1)
        address-locations [0 1]
        updated-rows (dtt/map-axis
                      (dtt/select C address-locations)
                      (fn [row] (f/+ row input-update))
                      1)]
    (doseq [l address-locations
            row updated-rows]
      (dtt/mset! C l row))
    C))

(comment
  (def word-length 1000)
  ;; needs to be large
  ;; (def address-count (long 1e6))
  (def A (->address-matrix word-length address-count 0.5))
  (def C (->content-matrix word-length address-count))
  (def T (into [] (repeatedly 100 #(->word word-length))))

  (def hamming-dists
    (into [] (for [test-data T]
               (let [a test-data
                     _ (def a a)
                     C (write! C (decode-addresses A a activation-radius-hamming) a)
                     output (read C (decode-addresses A a activation-radius-hamming))
                     ]
                 (hamming-dist a output)))))



  )
