(ns bennischwerdtner.hd.sdm-bseg
  (:require
   [bennischwerdtner.hd.binary-sparse-segmented :as hd]
   [clojure.test :as t]
   [tech.v3.datatype.functional :as f]
   [tech.v3.parallel.for :as pf]
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

(def activation-radius-hamming 447)
(def counter-range 15)

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
                                     segment-length
                                     )))]
                    indices)))))]
    (dtype/clone (dtt/compute-tensor
                   [hard-address-location-count
                    address-length]
                   (fn [i j] (if ((ones i) j) 1 0))
                   :int8))))


;; A is represented here as
;; each neuron in A is represented here as a
;; ( idx1, idx2, idx3, ...)
;;
;; (defn ->address-matrix-segmented
;;   [address-length segment-length hard-address-location-count
;;    address-density-k]
;;   (let [ones
;;         (into
;;          []
;;          (repeatedly
;;           hard-address-location-count
;;           (fn []
;;             ;; address-density-k ones, each in
;;             ;; one segment of addr x
;;             (into
;;              #{}
;;              (let [segment-count (/ address-length
;;                                     segment-length)
;;                    indices (repeatedly
;;                             segment-count
;;                             #(fm.rand/irand
;;                               segment-length))
;;                    indices
;;                    (dtt/->tensor (f/+ indices
;;                                       (f/* (range segment-count)
;;                                            segment-length)))
;;                    indices
;;                    (take address-density-k
;;                          (shuffle indices))]
;;                indices)))))]
;;     (dtype/clone
;;      (dtt/compute-tensor
;;       [hard-address-location-count
;;        address-length]
;;       (fn [i j] (if ((ones i) j) 1 0))
;;       :int8))



;;     ))



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
  [address-matrix address decoder-threshold]
  ;; d
  (let [inputs (dtt/reduce-axis
                address-matrix
                ;; this is where you gain massively
                ;; from parallelizing. Each neuron can
                ;; integrate and fire on it's own
                (fn [row] (f/dot-product row address)))
        activations (f/<= decoder-threshold inputs)]
    ;; (dtt/->tensor activations {:datatype :int8})
    ;; (dtype/clone activations)
    (unary-pred/bool-reader->indexes activations)))

(comment
  (time
   (decode-addresses A a decoder-threshold))

  )



;; (defn pdecode-addresses
;;   [address-matrix address decoder-threshold]
;;   ;; d
;;   (let [activations
;;           (pf/indexed-map-reduce
;;             (count address-matrix)
;;             (fn indexed-map-fn [start-idx group-len]
;;               (dtt/reduce-axis
;;                 (dtt/select address-matrix
;;                             (range start-idx
;;                                    (+ start-idx group-len)))
;;                 ;; this is where you gain massively
;;                 ;; from parallelizing. Each neuron can
;;                 ;; integrate and fire on it's own
;;                 (fn [row]
;;                   (f/<= decoder-threshold
;;                         (f/dot-product row address)))))
;;             ;; identity concat
;;             #(apply concat %)
;;             ;; (fn reduce-fn [values]
;;             ;;   ;; (dtt/clone (f/<= ;;
;;             ;;   decoder-threshold
;;             ;;   ;;             (apply concat values)))
;;             ;;   (f/<=
;;             ;;    decoder-threshold
;;             ;;    values)
;;             ;;   )
;;           )
;;         ;; (dtt/reduce-axis
;;         ;;  address-matrix ;; this is where you gain
;;         ;;  massively ;; from parallelizing. Each
;;         ;;  neuron can ;; integrate and fire on it's
;;         ;;  own
;;         ;;  (fn [row] (f/dot-product row address)))
;;         ;; activations (f/<= decoder-threshold inputs)
;;        ]
;;     (unary-pred/bool-reader->indexes activations)))


(defn pdecode-addresses
  [address-matrix address decoder-threshold]
  ;; d
  (let [activations
        (pf/indexed-map-reduce
         (count address-matrix)
         (fn indexed-map-fn [start-idx group-len]
           (unary-pred/bool-reader->indexes
            (dtt/reduce-axis
             (dtt/select address-matrix
                         (range start-idx
                                (+ start-idx group-len)))
             ;; this is where you gain massively
             ;; from parallelizing. Each neuron can
             ;; integrate and fire on it's own
             (fn [row]
               (f/<=
                decoder-threshold
                (f/dot-product row address))

               ;; (dtype-argops/argfilter
               ;;  decoder-threshold
               ;;  (f/dot-product row address))

               ))))

         ;; identity concat
         ;; #(apply concat %)
         ;; identity


         ;; (fn reduce-fn [values]
         ;;   ;; (dtt/clone (f/<= ;;
         ;;   decoder-threshold
         ;;   ;;             (apply concat values)))
         ;;   (f/<=
         ;;    decoder-threshold
         ;;    values)
         ;;   )

         identity

         ;; (reduce
         ;;  (fn [{:keys [idx activations]} inputs]
         ;;    {:activations (bitmap/reduce-union
         ;;                   [activations
         ;;                    (f/+ idx
         ;;                         (f/* (range (count
         ;;                                      inputs))
         ;;                              inputs))])
         ;;     :idx (+ idx (count inputs))})
         ;;  {:activations (bitmap/->bitmap) :idx 0}
         ;;  activations)

         )
        ;; (dtt/reduce-axis
        ;;  address-matrix ;; this is where you gain
        ;;  massively ;; from parallelizing. Each
        ;;  neuron can ;; integrate and fire on it's
        ;;  own
        ;;  (fn [row] (f/dot-product row address)))
        ;; activations (f/<= decoder-threshold inputs)
        ]
    ;; (def activations activations)
    ;; (unary-pred/bool-reader->indexes activations)

    ;; (:activations activations)

    activations))


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
;; addresses are skewed,
;; e.g. L = 100
;; N = 1000

(defn ->word [] (hd/->hv))

(comment
  (do
    (def word-length (:bsdc-seg/N hd/default-opts))
    (def address-length word-length)
    (def address-count 10000)
    (def address-density-k 5)
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






    )





  (def a (->word word-length word-density-l))
  ;; (decode-addresses A a address-density-k)
  (hamming-dist
   (read C (decode-addresses A a address-density-k))
   a)
  (def T
    (into []
          (map (fn [_] (->word word-length word-density-l)))
          (range 100)))




  (time
   (doseq
       [d T]
       (write! C (decode-addresses A d address-density-k) d)))

  (hamming-dist
   (read C (decode-addresses A (first T) address-density-k))
   (first T))

  (Math/pow 0.01 3)
  (Math/pow 0.01 3)

  ;; I I would have 1 dendrite per hd
  ;; that would be equivalent to p = 0.01,
  ;; it would be like throwing the dice for the 1 segment I connect to
  ;;
  (hd/hv->indices (hd/->hv))
  (Math/pow 0.01 2)

  ;; if you say 1/2 then you get 2 chances, +
  (+ 0.01 0.01)

  (* 1e-3 1e-3)

  (*
   (Math/pow 0.01 3)
   ))


;; this makes an 'intermediate' design

;; 5 mossy fibres, then say the threshold is 3
;; but this could be dynamic
;;
;; G - (G = 3) threshold for a golgi cell (addr. decoder)
;; k - (k = 5) nr. of golgi dendrites (nr. of 1s in the addr. vector)
;;     Note in the segmented address (x) design, there should be maximally one 1 per segment
;; L - (L = 100, 1 per segment) the number of ones in the address
;;


(comment



  ;; p
  ;; l = 100
  ;;
  (let [l 100]
    (Math/pow (/ l word-length) address-density-k))
  ;; chose address-density-k to get p ~=
  ;;
  ;;

  (let [M address-count
        ;; dataset count
        T 10000
        p-ideal (Math/pow (* 2 M T) (/ -1 3))]
    p-ideal)
  0.0017099759466766976


  (defn factorial [n]
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

  )


(comment









  (time
   (let [wr (dtype/alloc-zeros :int8 address-count)]
     (reduce
      (fn [{:keys [idx activations]} inputs]
        (doseq [idx (dtype-argops/argfilter #(< 0 %) inputs)]
          (dtype/set-value! wr idx 1))
        {:idx (+ idx (count inputs))})
      {:activations (bitmap/->bitmap) :idx 0}
      activations)
     (dtt/->tensor wr)))

  (time
   (reduce (fn [{:keys [idx activations out]} inputs]
             ;; (doseq [idx (dtype-argops/argfilter
             ;; #(< 0 %) inputs)]
             ;;   (dtype/set-value! wr idx 1))
             {:idx (+ idx (count inputs))
              ;; :activations (conj )
              :out (conj out
                         (unary-pred/bool-reader->indexes
                          inputs))})
           {:idx 0 :out []}
           activations))







  )
