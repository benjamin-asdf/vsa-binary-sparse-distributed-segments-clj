(ns bennischwerdtner.hd.sdm-bseg
  (:require
   [bennischwerdtner.hd.binary-sparse-segmented :as hd]
   [clojure.test :as t]
   [tech.v3.datatype.functional :as f]
   [tech.v3.parallel.for :as pf]
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

;; But here, I make an example of binary sparse segmented (hypervector) words

(def counter-range [0 15])

;;
;; address-density is 0.5 here, this is sparse in hyperplane design (Jaeckle),
;; where k = 10, N = 1000, p = 0.5^k, p = 0.001
;;


;; in hyperplane design, each row has 3 (k = 3) ones and (N - k = 997) zeros

;; In Cerebellum, this is ~5 per granule cell,
;; motivating the idea to make this sparse
;;

(defn ->address-matrix-segmented
  [address-length segment-length hard-address-location-count
   address-density-k]
  (let [ones
          (into
            []
            (repeatedly
              hard-address-location-count
              (fn []
                ;; address-density-k ones, each in
                ;; one segment of addr x
                (into
                  #{}
                  (let [segment-count (/ address-length
                                         segment-length)
                        indices (repeatedly
                                  segment-count
                                  #(fm.rand/irand
                                     segment-length))
                        indices
                          (dtt/->tensor
                            (f/+ indices
                                 (f/* (range segment-count)
                                      segment-length)))
                        indices (dtt/select
                                  indices
                                  (take
                                    address-density-k
                                    (dtype-argops/argshuffle
                                      segment-length)))]
                    indices)))))]
    (dtype/clone (dtt/compute-tensor
                   [hard-address-location-count
                    address-length]
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
;; In an intermediate design, like here, you count up to decoder-threshold G.
;; (G = 2, k = 5)
;;
;; G - (G = 2) threshold for a golgi cell (addr. decoder)
;; k - (k = 5) nr. of golgi dendrites (nr. of 1s in the addr. vector)
;;     Note in the segmented address (x) design, there should be maximally one 1 per segment
;; L - (L = 100, 1 per segment) the number of ones in the address
;;

(defn decode-addresses
  [address-matrix address decoder-threshold]
  ;; d
  (let [inputs (dtt/reduce-axis
                address-matrix
                ;; this is where you will gain
                ;; massively from parallelizing. Each
                ;; neuron can integrate and fire on
                ;; it's own
                (fn [row] (f/dot-product row address)))
        activations (f/<= decoder-threshold inputs)]
    (unary-pred/bool-reader->indexes activations)))

(comment
  ;; this is 'slow'
  (time
   (decode-addresses A a decoder-threshold))

  )


;; --------------------------
;; Storage
;; --------------------------
;;
;; The content matrix:
;;
;; counter in range c (c = 0...15).
;;
;; At each write, increment the counter at the bit location of all non-zero bits
;; of the input word.
;;
;;
;; In this sparse segmented version:
;;
;; For reading, instead of counting +/-, take 1 on-bit per segment,
;; the one with the highest activation.
;; With a strategy for breaking ties that is allowed to be random.
;;
;; It is biologically plausibly implemented via inhibitory interneurons, creating 'sheet inhibition'.
;; Observe that this requires the postulated interneurons to inhibit Purkinje cells
;; - indiscrimitely
;; - faster than granule cell parallel fiber input
;; - proportional to granule cell input
;;

(defn ->content-matrix
  [word-length hard-address-location-count]
  (dtt/reshape
   (dtt/->tensor
    (dtype/alloc-zeros
     :int8
     (*
      hard-address-location-count
      word-length)))
   [hard-address-location-count word-length]))


;; Returns the updated content matrix
;; In this version, only increment the bit locations

(defn write!
  [content-matrix address-locations input-word]
  (let [C content-matrix
        input-update input-word
        updated-rows (dtt/map-axis
                       (dtt/select C address-locations)
                       (fn [row]
                         (-> (f/+ row input-update)
                             ;; (f/max (first
                             ;; counter-range))
                             (f/min (second
                                      counter-range))))
                       1)]
    (doseq [l address-locations
            row updated-rows]
      (dtt/mset! C l row))
    C))

(comment
  (write! (dtt/->tensor [[0 0 0] [0 0 0]]) [0] [0 0 1])
  ;; [[0 0 1]
  ;;  [0 0 0]]
  )


;; 1. sum up the address location bits of activated rows
;; 2. For each segment of the output word, take the highest bit
;;
;; Each of the bits in the output models 1 Purkinje cell (output cell)
;;


;;
;;
;;
;; activations
;; (granule cells)
;; |- golgi cells inhibit
;;
;;
;;
;;
;;  +--+        content matrix:
;;  |0 |       +-------------+------------------------+
;;  |1 +------>| 1 0 1 0 2 0 |                        | parallel fibres
;;  |0 |       |             |                        |
;;  |. |       |             |                        |                |-(stellate cells inhibit)
;;  |1 +------>| 1 1 0 1 1 0 |                        |
;;  |  |       +---------+---+------------------------+
;;  +--+                 |
;;   y                   v
;;              +------------+-------------------------+
;;              |        3   |                         | sum          ('purkinje cell inputs')
;;              +------------+-------------------------+              |-(basket cells inhibit)
;;
;;                           1,                  ...   segment-count
;;
;;
;;              +------------+-------------------------+
;;              |        1   |                         | Z        <---+
;;              +------------+-------------------------+ word out     |
;;                                                                    |
;;                                                                    |
;;               one non-zero bit per segment                         |
;;                                                        we can imagine purnkinje cell activations,
;;                                                        but the reader neuron of purkinje cells
;;                                                        might modify the word-out
;;
;;                                                        deep cerebellar nucleus
;;
;;
;;
;;
;; - In cerebellum, purkinje cells are inhibitory
;; - Climbing fiber + parallel fiber input leads to *long term depression*
;; - So it would be like decrementing the bits in content matrix, and reading the output flipped
;;
;;
;;
;;

;; - each parallel fibers goes through 500 purkinje cells
;; - each purkinje cell recieves information from 200k pfs
;; - you need large numbers of granule cell inputs to make a purkinje fire
;; -
;; - Buzsáki: This arrangment suggests that cerebellar "modules", which roughly
;; correspond to the extend of parallel fibers, process the incomming inputs locally,
;; but they do not need to consult or inform the rest to the cerebellum about the locally
;; derived computation.
;;
;;






;; (defn sdm-read
;;   [content-matrix address-locations word-lenght
;;    word-segment-length word-segment-count]
;;   ;; depends on U (the width of the content matrix) ==
;;   ;; word-length
;;   (let [indices (-> (dtt/reduce-axis (dtt/select
;;                                       content-matrix
;;                                       address-locations)
;;                                      f/sum
;;                                      0)
;;                     (dtt/reshape [word-segment-count
;;                                   word-segment-length])
;;                     (dtt/reduce-axis dtype-argops/argmax))]
;;     ;; create a sparse binary segmented hypervector
;;     ;; from the indices
;;     (hd/indices->hv indices)))

;; N = 10000 (word length).
;; L = 100 (word non zero bits count).
;; segment-length = 100 (words are segmented, 1 non-zero bit per segment).

(defn ->word [] (hd/->hv))

(defn auto-associate!
  [C A input-word decoder-threshold]
  (write!
   C
   (decode-addresses A input-word decoder-threshold)
   input-word))

(comment
  (def word-length (:bsdc-seg/N hd/default-opts))
  (def address-length word-length)
  (def address-count (long 1e4))
  (def address-density-k 6)
  ;; G
  (def decoder-threshold 2)

  (def A
    (->address-matrix-segmented address-length
                                (:bsdc-seg/segment-length
                                 hd/default-opts)
                                address-count
                                address-density-k))
  (def a (->word))
  (time (decode-addresses A a decoder-threshold))

  (def C (->content-matrix word-length address-count))
  ;; (decode-addresses A a address-density-k)

  (time (auto-associate! C A a decoder-threshold))
  (write!
   C
   (decode-addresses A a decoder-threshold)
   a)

  (hd/similarity
   a
   (sdm-read C
         (decode-addresses A a decoder-threshold)
         word-length
         (:bsdc-seg/segment-length hd/default-opts)
         (:bsdc-seg/segment-count hd/default-opts)))
  1.0

  ;; (def T (into [] (map (fn [_] (->word))) (range 1000)))
  (def T (into [] (map (fn [_] (->word))) (range 100)))

  (time
   (doseq
       [d T]
       (auto-associate! C A d decoder-threshold)))

  (doall
   (for
       [d T]
       (let [sim
             (hd/similarity
              d
              (sdm-read C
                    (decode-addresses A d decoder-threshold)
                    word-length
                    (:bsdc-seg/segment-length hd/default-opts)
                    (:bsdc-seg/segment-count hd/default-opts)))]
         (when (< sim 1.0)
           (def d d))
         sim)))

  [(decode-addresses A d decoder-threshold)
   (decode-addresses A
                     (hd/thin (hd/bundle d (hd/->seed)))
                     decoder-threshold)]

  [[144 947 1905 3413 3591 3720 4000 5677 6851 7242 9033 9095]
   [763 1464 1822 1843 2036 2191 3720 4239 6204 6585 6851 7123 7373 7814 9033 9095 9914]]

  (let [d-prime
        (hd/thin (hd/bundle d (hd/->seed)))
        a2 (read-word d-prime)]
    (def d-prime d-prime)
    (hd/similarity d (read-word a2)))

  (defn read-word
    [a]
    (sdm-read C
          (decode-addresses A a decoder-threshold)
          word-length
          (:bsdc-seg/segment-length hd/default-opts)
          (:bsdc-seg/segment-count hd/default-opts))))

(comment

  (clojure.set/intersection
   (into #{} (decode-addresses A a decoder-threshold))
   (into #{} (decode-addresses A c decoder-threshold)))

  (clojure.set/intersection
   (into #{} (decode-addresses A a decoder-threshold))
   (into #{} (decode-addresses A b decoder-threshold)))

  (def a (hd/->seed))
  (def b (hd/->seed))
  (def c (hd/thin (hd/bundle a b)))

  (do
    (auto-associate! C A a decoder-threshold)
    (auto-associate! C A b decoder-threshold)
    (auto-associate! C A c decoder-threshold))


  ;; this version doesn't have the critical distance property

  [(hd/similarity b (read-word b))
   (hd/similarity c (read-word c))
   (hd/similarity c (read-word (read-word c)))
   (hd/similarity a (read-word c))
   (hd/similarity b (read-word c))]

  [1.0 0.77 0.83 0.19])

(comment

  ;;
  ;; Appendix: Address decoder probability
  ;;

  (let [M address-count
        ;; dataset count
        T 10000
        p-ideal (Math/pow (* 2 M T) (/ -1 3))]
    p-ideal)
  0.0017099759466766976

  (defn
    (reduce * (range 1 (inc n))))

  (defn choose [n k]
    (/ (factorial n) (* (factorial k) (factorial (- n k)))))

  (defn dice-throw-probability
    [total-throws success-throws dice-sides success-side]
    (let [p (Math/pow (/ success-side dice-sides) success-throws)
          q (Math/pow (/ (- dice-sides success-side) dice-sides) (- total-throws success-throws))
          combinations (choose total-throws success-throws)]
      (* combinations p q)))



  (dice-throw-probability 5 2 100 1)
  ;; 9.702990000000001E-4
  ;; -> from this I conclude that 2/5 is a good values for G/k.


  (dice-throw-probability 5 3 100 1)
  ;; 3/5 is ~100x to small probability
  ;; 9.801000000000002E-6
  ;;
  ;; 1/5 is ~48x to much
  (dice-throw-probability 5 1 100 1)
  ;; 0.0480298005

  ;; ... somebody more theoretical can do the math


  (dice-throw-probability 7 2 100 1)
  0.00199707910479
  ;; maybe k = 6 is better
  (dice-throw-probability 6 2 100 1)
  0.001440894015

  )
