(ns bennischwerdtner.sdm.sdm
  (:require
   [clojure.set :as set]
   [bennischwerdtner.pyutils :as pyutils :refer [*torch-device*]]
   [bennischwerdtner.hd.binary-sparse-segmented :as hd]
   [tech.v3.datatype.functional :as f]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [tech.v3.datatype.bitmap :as bitmap]
   [fastmath.random :as fm.rand]
   [fastmath.core :as fm]
   [tech.v3.datatype.unary-pred :as unary-pred]
   [tech.v3.datatype.argops :as dtype-argops]
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python.ffi :as py-ffi]
   [libpython-clj2.python :refer [py. py.. py.-] :as py]))

;;
;; This is a sparse distributed memory for binary sparse segmented hypervectors
;;
;;
;;

;; Adapted from Kanerva 1993[1]:
;; And mapping to neuroanatomy of cerebellum
;;
;;
;;
;;         ADDRESS REGISTER                             WORD IN REGISTER
;;
;;            N = 10.000                                 U = N = 10.000
;;        +--------------+                           +------------------+
;;      x |              |                        w  | 1  |1  |   1  |  |  (sparse, segmented)
;;        +------+-------+                           +----------+-------+
;;               |                                              |
;;               |                                              |
;;               |      N      d               y                |      U
;;        +------v-------+    +-+             +-+    +----------v-------+
;;        |              |    |3|             |1+---->                  |
;;        |              |    |0|             |0|    |                  |
;;        |      A       +--> |2| --------->  |1|---->        C         |
;;        | M hard addr. |    |0|  threshold  |0|    |   M x U counters |
;;      M |              |    |1|             |0|  M |                  |
;;        +--------------+    +-+             +-+    +---------+--------+
;;                                                             |
;;                                             |               |
;;                             |               |               v
;;                             |               |      +-----------------+
;;                             |               |   S  |                 |  sums
;;                             v               |      +--+-----+----+---+
;;                        address overlap      |         |     |    |
;;                         ~d of [1,2]         |         |     |    |
;;                                             |         v     v    v        (where S >= read-threshold)
;;                         activations    <----+       --+-----+----+----
;;                         ( 2 <= d)                     v     v    v        top-k per segment
;;                                                    +-----+----+------+
;;                                                 z  |     |    |      |
;;                                                    +-----+----+------+
;;                                                      s1    s2, ... segment-count
;;
;;
;;                                                    word out register
;;
;;                                                   10.000 bits = segment-count * segment-length
;;
;;
;;
;; x - address word                        'mossy fiber input'
;; A - Address Matrix                      'mossy fibers -> granule cells synapses'
;; M - hard-locations-count                'granule cell count'
;; y - address-activations                 'granule cell activations'
;; w - input word                          'climbing fiber input'
;; C - Content Matrix                      'parallel fibers -> purkinje synapses'
;; S - content sums                        'purkinje inputs'
;; z - output word                         'purkinje activations', or downstream purkinje reader
;;
;; -----------------
;; Address decoder
;; -----------------
;;
;; - M hard address locations of N (address-word-lenght) width
;; - address locations are sparse 0s and 1s with address-density << 1
;; - Each address location models one granule cell, granule cells outnumber mossy fibers (their inputs) 200 to 1
;;   (https://en.wikipedia.org/wiki/Cerebellar_granule_cell#:~:text=Granule%20cells%20receive%20all%20of,a%20much%20more%20expansive%20way.)
;;
;;
;; Since addreses are not dense, this is a 'intermediate design' (Jaeckel, L.A. 1989b)
;;
;; Address Decoding
;; -----------------
;; - let `address` (query) be `x`
;; - find the overlap for each row in `A`, that is a vector of overlaps `d` (~ golgi inputs)
;; - cutoff with `decoder-threshold` (e.g. decoder-threshold = 2),
;;    - this is the address location activation vector `y`
;; - this is a tensor of size [M], with a tiny fraction non zero, e.g. 36 out of 10.000
;; - (y ~= golgi cell activations)
;; - subsequently, either read or write using active locations.
;;
;; In the address decoder step, we massively benefit from parallelism.
;;
;; -----------------
;; Storage / Content Matrix `C`
;; -----------------
;;
;; - M x U counters
;; - here, U == N, allowing for auto association. I.e. address-word == input-word.
;; - Different from Kanerva 1993, where each location is in range {-15...15},
;;   here range is {0..`counter-max`}, counter-max = 100 (?)
;;
;; Read (y):
;; - sum up counters of the y activated address content locations of `C`, sums `S` ~ purkinje inputs.
;; - Optionally (not here): Remove sums below read-threshold, reducing noise, trading signal strenght. In other words, this increases
;;   the count and strength of address locations for a stored word `a` to be retrieved.
;; - The bit count active is a meassure of the `confidence` for the query relating to a stored word.
;; - In Kanerva 1993: take the sign of the sums, output vector `z` elements are in {-1,1}
;; - Here, take top-k (`read-k`) non zero bits for each word segment, to accomodate a binary segmented sparse design.
;; - output vector `z` (size N), has read-k * segment-count non zero bits, `z` elements are in {0,1}, where the count of non-zero is << N
;; - iff `read-k` == 1, then `z` is maximally sparse [[hd/maximally-sparse?]]
;;
;;
;;
;; Write (y, input-word):
;;
;; - For each active location, increment the in C where input-word has a non-zero bit
;; - clamp C to the counter range
;; - this is flipped from cerebellum where mossy fiber + parallel fiber input makes LTD on the synapse (i.e. it decrements the weight),
;;   presumably, this flips back by the purkinje being inhibitory.
;;   The reason for this flipped arrangement remains elusive.
;;   The answer lies with the microcircuits of purkinje readers in deep cerebellar nuclei.
;;   Presumably, you would find a reason for inhibitory inputs to be more useful.
;;

;; ========================
;; Parameters
;; M - memory count (e.g. 1e6)
;; T - (data set) should be 1-5% of M
;; p - probability of activation, ideally p = 0.000368 (depends on M and T)
;; This is important, number of hard locations activated for an input
;; The best p maximizes signal to noise, is approx. 2MT^-1/3
;;
;;

(comment
  ;; what is p?
  ;; depends on the M the hard locations count,
  ;; density of the address matrix
  ;; density of address words
  ;; and decoder threshold
  ;;
  (require-python '[numpy.random :as nprandom])

  (defn calculate-activation-probability [address-word-length address-density word-density decoder-threshold num-samples]
    (let [p-match (* address-density word-density)
          samples (nprandom/binomial address-word-length p-match num-samples)]
      (float (/ (np/sum (np/greater_equal samples decoder-threshold)) num-samples))))

  (defn ideal-p [dataset-count address-count]
    (Math/pow (* 2 dataset-count address-count) (- (/ 1 3))))

  (ideal-p 1e3 1e6)

  ;; 0.003872

  ;; for
  ;; T = 1.000
  ;; M = 10.000
  ;; address density = 0.0009, decoder threshold = 2
  ;; seem to be in a good balpark
  ;; (unless the calculation is wrong)

  ;; ==>
  ;; One can simply setup an address decoder and empircally play until one finds a good config
  ;; where good means the address-locations decoded are roughly p = (address-locations / M) = idea-p = 2MT^-1/3
  ;;


  ;;
  ;; I do this by playing around with such numbers:
  ;;
  (torch/sum
   (decode-address (->decoder-coo
                    {:address-count (long 1e6)
                     :address-density 0.00001
                     :word-length (long 1e4)})
                   (hd/->hv)
                   1))
  ;; tensor(187, device='cuda:0')
  ;; (/ 200 1e6)
  ;; 2.0E-4
  ;; (ideal-p 1e3 1e6)
  ;; 7.937005259841001E-4
  )


;; ----------------------
;; Concrete torch implementation
;; ----------------------
;;
;; The following are 2 concrete implementations of the SDM.
;;
;; The first is a dense implementation, where the programatic structure is hopefully easier to follow.
;;
;; The second is a torch.sparse 'coo' implementation, where the semantic is supposed to be the same.
;;

(do
  ;;
  ;; Anything backed by a :native-buffer has a zero
  ;; copy pathway to and from numpy.
  ;; Https://clj-python.github.io/libpython-clj/Usage.html
  (alter-var-root #'hd/default-opts
                  (fn [m]
                    (assoc m
                           :tensor-opts {:container-type
                                         :native-heap})))
  (require-python '[numpy :as np])
  (require-python '[torch :as torch])
  (require-python '[torch.sparse :as torch.sparse])
  (require '[libpython-clj2.python.np-array]))

(def counter-max 15)

;; https://pytorch.org/docs/stable/sparse.html#sparse-csr-tensor
;; The primary advantage of the CSR format over the COO format is better use of storage and much faster computation operations such as sparse matrix-vector multiplication using MKL and MAGMA backends.

(defn ->address-matrix
  [address-count address-length density]
  (py.. (torch/less (torch/rand [address-count
                                 address-length]
                                :device
                                *torch-device*)
                    density)
    (to torch/float16)))

;; kinda funny that many rows are never activated
;; in lieu of the amount of granule cells, you can wonder
;;
;; outcome here has less density than advertised because random indices overlap
;;
(defn ->address-matrix-coo
  [address-count word-length density]
  (let [nse (long (* address-count word-length density))
        row-indices (torch/randint 0
                                   address-count
                                   [nse]
                                   :device
                                   *torch-device*)
        col-indices (torch/randint 0
                                   word-length
                                   [nse]
                                   :device
                                   *torch-device*)
        i (torch/stack [row-indices col-indices])
        v (torch/ones [nse]
                      :dtype torch/float32
                      :device *torch-device*)
        A (py.. (torch/sparse_coo_tensor i
                                         v
                                         :size
                                         [address-count
                                          word-length])
                (coalesce))
        _ (py.. A (values) (clamp_ 0 1))]
    A))

(defn ->address-locations
  [address-count indices]
  (py/set-item! (torch/zeros [address-count]
                             :dtype torch/bool
                             :device *torch-device*)
                indices
                true))

(comment
  (->address-locations 10 (torch/tensor [1 2 3] :dtype torch/long :device *torch-device*)))

(defn ->content-matrix
  [address-count word-length]
  (torch/zeros [address-count word-length]
               :device
               *torch-device*))

(defn ->content-matrix-coo
  [address-count word-length]
  (torch/sparse_coo_tensor :size [address-count word-length]
                           :device *torch-device*
                           :dtype torch/uint8))



;; -----------------------------
;; Address Decoding
;; -----------------------------

;;
;;         ADDRESS REGISTER
;;
;;            N = 10.000
;;        +--------------+
;;      x |              |
;;        +------+-------+
;;               |
;;               |
;;               |      N      d               y
;;        +------v-------+    +-+             +-+
;;        |              |    |3|             |1+
;;        |              |    |0|             |0|
;;        |      A       +--> |2| --------->  |1|
;;        | M hard addr. |    |0|  threshold  |0|
;;      M |              |    |1|             |0|
;;        +--------------+    +-+             +-+
;;
;;                                             |
;;                             |               |
;;                             |               |
;;                             |               |
;;                             v               |
;;                        address overlap      |
;;                         ~d of [1,2]         |
;;                                             |
;;                         activations    <----+
;;                         ( 2 <= d)
;;
;;
;;
(defn decode-addresses
  [address-matrix address decoder-threshold]
  ;; d
  (let [address (py.. (pyutils/ensure-torch address)
                  (to :dtype torch/float16))
        inputs (torch/mv address-matrix address)
        activations (torch/ge inputs decoder-threshold)]
    ;; y
    activations))

(defn decode-addresses-coo
  [address-matrix address decoder-threshold]
  (let [out (torch/zeros [(py.. address-matrix (size 0))]
                         :dtype torch/bool
                         :device *torch-device*)]
    (py/with-gil-stack-rc-context
      (let [address (pyutils/ensure-torch address)
            inputs (torch/mv address-matrix address)
            activations (torch/ge inputs
                                  (torch/tensor
                                   decoder-threshold
                                   :device *torch-device*
                                   :dtype torch/float32))]
        (py.. out (copy_ activations))
        out))))

(comment
  (decode-addresses
   (->address-matrix 3 3 0)
   (torch/tensor [0 1 1] :dtype torch/float16 :device *torch-device*)
   1)

  (decode-addresses-coo
   (->address-matrix-coo 3 3 0)
   (torch/tensor [0 1 1] :dtype torch/float32 :device *torch-device*)
   1))


;; ---------------------------------
;; Write
;; ---------------------------------

;;
;;               WORD IN REGISTER
;;
;;                U = N = 10.000
;;            +------------------+
;;         w  | 1  |1  |   1  |  |  (sparse, segmented)
;;            +----------+-------+
;;                       |
;;                       |
;;      y                |      U
;;     +-+    +----------v-------+
;;     |1+---->          +1      |
;;     |0|    |                  |
;;     |1|---->        C         |
;;     |0|    |   M x U counters |
;;     |0|  M |                  |
;;     +-+    +------------------+
;;
;; - increment where input and address activations meet
;; - clamp to [[counter-max]]

(defn write!
  [content-matrix address-locations input-word]
  (let [input-word (pyutils/ensure-torch input-word)]
    (py/set-item! content-matrix
                  address-locations
                  (py.. (py/get-item content-matrix
                                     address-locations)
                        (add_ input-word)
                        (clamp_ :min 0 :max counter-max)))))

(defn write-coo!
  [content-matrix address-locations input-word]
  (py/with-gil-stack-rc-context
    (let [;; btw, this causes host-device
          ;; synchronization
          ;; (doesn't matter here yet because I have
          ;; datatransfers all the time anyway)
          activated-locations (py.. (torch/nonzero
                                      address-locations)
                                    (view -1))
          word-nonzero (py.. (torch/nonzero
                               (pyutils/ensure-torch
                                 input-word))
                             (view -1))
          indices (py.. (torch/cartesian_prod
                          activated-locations
                          word-nonzero)
                        (t))
          values (torch/ones (py.. indices (size 1))
                             :dtype torch/uint8
                             :device *torch-device*)
          update (torch/sparse_coo_tensor
                   indices
                   values
                   (py.. content-matrix size))
          _content-matrix
            (py.. content-matrix (add_ update) (coalesce))]
      (py.. _content-matrix values (clamp_ 0 counter-max))
      content-matrix)))


(defn read-coo-1
  "
  Returns the `sums` for each counter column in `content-matrix`,
  selecting the activated `address-locations` rows.


        +---+
        |   |                       address-locations
      +-+---+------------------+           |
 ->   | | 2 |                  | on        |
      +-+---+------------------+           |
      | |   |                  | off       |
      +-+---+------------------+           |
 ->   | | 8 |                  | on        v
      +-+---+------------------+ ...
        |   |
        ++--+ ...                C  address-count x word-length
         |
         |  ^
         |  +---------------------- activated counters
         |
         |
         |
         |
         v
     +------+-------------------+
  S  |  10  | 0  11 , ....      | sums size [word-lenght]
     +------+-------------------+
                                  per columm



  "
  [content-matrix address-locations]
  (let [addr-indices (torch/squeeze (torch/nonzero
                                     address-locations)
                                    1)
        sums (py.. (torch/sum (torch/index_select
                               content-matrix
                               0
                               addr-indices)
                              0)
               (to_dense))]
    sums))

(defn ensure-cpu [tens]
  (py.. tens (to "cpu")))
(defn torch->numpy [tens]
  (py.. tens (numpy)))
(defn torch->jvm
  [torch-tensor]
  (-> torch-tensor
      ensure-cpu
      torch->numpy
      dtt/ensure-tensor))
(defn ensure-jvm [tens]
  (if (dtt/tensor? tens)
    tens
    (torch->jvm tens)))

(defn sdm-read-coo
  "Returns a lookup result for the address-actions `address-locations`, and a confidence value.

  `result`: A binary segmented hypervector reading from `address-locations`.
  `confidence`: The normalized sum of storage counters that contribute to the result.

  This repesents an approximation of the confidence of the result being an item in memory.
  If `top-k` > 1 this would increase accordingly.
  This can exceed 1, if counter locations are higher than 1, then an item was stored multiple times.

  If close to one, the confidence is high. If close to zero, the confidence is low.

  `top-k`: number of non zero bits to take from each segment.

  If `top-k` == 1, the output is usually maximally sparse.

 y - address activations

 +--+        +---+----------------------+
 |  +------->|   |                      |
 |  |        +---+----------------------+
 |  |        |   |                      |
 |  |        |   |                      |
 |  |        |   |                      |
 |  |        |   |      C               |
 |  |        |   |                      |
 +--+        +---+----------------------+
  y            s0| s1 , ... segment-count
                 |
                 | SUM
                 |
                 v
             +-------+------+-----------+
             |   s0  |      |           |
             +-------+------+-----------+
              segment1, ....

             top-k per segment       s0,s1,... the summed counter 'winning' values for each segment.
                                     The bit counter content amount contributing to the result is a confident meassure.
                 |
                 |
                 v
             +--------------------------+
             |                          |
             +--------------------------+

               output hdv

              top-k non zero bits per segment


  See [[hd/maximally-sparse?]].
  "
  ([content-matrix address-locations top-k]
   (sdm-read-coo content-matrix
                 address-locations
                 top-k
                 hd/default-opts))
  ([content-matrix address-locations top-k
    {:bsdc-seg/keys [segment-count segment-length N]}]
   (let [out (torch/zeros [N] :device *torch-device*)]
     ;; you must be careful to not let python objects
     ;; escape the rc-context, this is the reason why
     ;; we do this datatransfer at the end,
     ;; (it's a very fast operation gpu->gpu)
     (py/with-gil-stack-rc-context
       (let [s (read-coo-1 content-matrix address-locations)
             topk-result
             (-> s
                 (torch/reshape [segment-count
                                 segment-length])
                 (torch/topk (min top-k segment-length)))
             result (torch/scatter
                     (torch/zeros [segment-count
                                   segment-length]
                                  :device
                                  *torch-device*)
                     1
                     (py.. topk-result -indices)
                     1)
             address-location-count
             (py.. (torch/sum address-locations) item)]
         {:address-location-count address-location-count
          :confidence (if (zero? address-location-count)
                        0
                        (py.. (torch/div
                               (torch/sum (py.. topk-result
                                                -values))
                               ;; also divide by top-k?
                               ;;
                               ;; scaling this with the
                               ;; address-locations
                               ;; count is probably
                               ;; taste. It turns out
                               ;; to say high
                               ;; confidence, if the
                               ;; count of address is
                               ;; low, when the address
                               ;; is a
                               ;; 'weak' (sub sparsity)
                               ;; hypervector. The
                               ;; caller would have to
                               ;; take this into
                               ;; account themselves.
                               ;;
                               (* segment-count address-location-count))
                              item))
          :result
          (do
            ;; you don't need to synchronize cuda
            ;; here apparently
            (py.. out (copy_ (torch/reshape result [N])))
            out)})))))

(defn sdm-read
  "See [[sdm-read-coo]]"
  ([content-matrix address-locations top-k]
   (sdm-read content-matrix
             address-locations
             top-k
             hd/default-opts))
  ([content-matrix address-locations top-k
    {:bsdc-seg/keys [segment-count segment-length N]}]
   (let [s (torch/sum (py/get-item content-matrix
                                   address-locations)
                      :dim
                      0)
         topk-result (-> s
                         (torch/reshape [segment-count
                                         segment-length])
                         (torch/topk (min top-k
                                          segment-length)))
         result (torch/scatter (torch/zeros
                                 [segment-count
                                  segment-length]
                                 :dtype torch/uint8
                                 :device *torch-device*)
                               1
                               (py.. topk-result -indices)
                               1)
         address-location-count (py.. (torch/nonzero
                                        address-locations)
                                      (size 0))]
     {:address-location-count address-location-count
      :confidence
        (if (zero? address-location-count)
          0
          (py.. (torch/div
                  (torch/sum (py.. topk-result -values))
                  (* segment-count address-location-count))
                item))
      :result (torch/reshape result [N])})))




;; not sure yet
;; these interfaces are just exploratory here
(defprotocol AddressDecoder
  (decode-address [this address decoder-threshold]))

(defprotocol SDMStorage
  (lookup-1
    [this address-locations top-k opts]
    [this address-locations top-k])
  (write-1 [this address-locations content]))

(defprotocol SDM
  (known? [this address]
          [this address decoder-threshold])
  (lookup [this address top-k]
          [this address top-k decoder-threshold])
  (converged-lookup [this address top-k]
                    [this address top-k decoder-threshold])
  (write [this address content]
    [this address content decoder-threshold]))

(defn ->decoder-coo
  [{:keys [address-count word-length address-density]}]
  (let [address-matrix (->address-matrix-coo
                         address-count
                         word-length
                         address-density)]
    (reify
      AddressDecoder
        (decode-address [this address decoder-threshold]
          (decode-addresses-coo address-matrix
                                address
                                decoder-threshold)))))


;;
;; 'dense' refers to the underlying torch backend
;; dense is easier to implement, serving as a reference implementation
;;
(defn dense-sdm
  [{:keys [address-count word-length address-density]}]
  (let [content-matrix (->content-matrix address-count
                                         word-length)
        address-matrix (->address-matrix address-count
                                         word-length
                                         address-density)]
    (reify
      AddressDecoder
        (decode-address [this address decoder-threshold]
          (decode-addresses
           address-matrix
           address
           decoder-threshold))
      SDMStorage
        (write-1 [this address-locations content]
          (write! content-matrix address-locations content))
        (lookup-1 [this address-locations top-k]
          (sdm-read content-matrix address-locations top-k))
      SDM
        (write [this address content decoder-threshold]
          (write-1
            this
            (decode-address this address decoder-threshold)
            content))
        (lookup [this address top-k decoder-threshold]
          (lookup-1
            this
            (decode-address this address decoder-threshold)
            top-k)))))

(defn ->sdm-storage-coo
  [{:keys [address-count word-length address-density]}]
  (let [content-matrix (->content-matrix-coo address-count
                                             word-length)]
    (reify
      SDMStorage
        (write-1 [this address-locations content]
          (write-coo! content-matrix
                      address-locations
                      content))
      (lookup-1
          [this address-locations top-k]
          (sdm-read-coo content-matrix
                        address-locations
                        top-k))
      (lookup-1
          [this address-locations top-k opts]
          (sdm-read-coo content-matrix
                        address-locations
                        top-k
                        opts)))))


;; Not yet figured out.
;; is the same as 'iterative' lookup
;; same for sequences and auto associative converge
(defn converged-lookup-impl
  [sdm address
   {:keys [stop? top-k decoder-threshold max-steps]}]
  (reduce
    (fn [{:as acc :keys [address result-xs]} n]
      (let [next-outcome
              (lookup sdm address top-k decoder-threshold)
            {:keys [stop-reason success?]}
              (stop? acc next-outcome)]
        (if stop-reason
          (cond (not success?) (ensure-reduced
                                 (assoc acc
                                   :stop-reason stop-reason
                                   :stop-result
                                     next-outcome))
                success?
                  (ensure-reduced
                    (assoc acc
                      :result-xs (conj result-xs
                                       next-outcome)
                      :success? true
                      :stop-reason stop-reason
                      :result-address (:result next-outcome)
                      :address (:result next-outcome))))
          {:address (:result next-outcome)
           :n n
           :result-xs (conj result-xs next-outcome)})))
    {:address address
     :result-xs [{:input? true :result address}]}
    ;; taste
    (range (or max-steps 7))))

(defn sparse-sdm
  [{:as opts
    :keys [address-count word-length address-density
           decoder]}]
  (let [decoder (or decoder (->decoder-coo opts))
        storage (->sdm-storage-coo opts)]
    (reify
      SDM
      (write [this address content decoder-threshold]
        (write-1 storage
                 (decode-address decoder
                                 address
                                 decoder-threshold)
                 content))
      (lookup [this address top-k decoder-threshold]
        (lookup-1 storage
                  (decode-address decoder
                                  address
                                  decoder-threshold)
                  top-k)))))

(def ->sdm sparse-sdm)







;; --------------------
;;  Unit Tests
;; --------------------
(comment
  (do
    (alter-var-root
     #'hd/default-opts
     #(merge %
             (let [dimensions 25
                   segment-count 5]
               {:bsdc-seg/N dimensions
                :bsdc-seg/segment-count segment-count
                :bsdc-seg/segment-length
                (/ dimensions segment-count)})))
    (let [r [(binding [*torch-device* :cpu]
               (let [m (dense-sdm {:address-count 100
                                   :address-density 0.2
                                   :word-length 25})
                     d (hd/->hv)]
                 (write m d d 2)
                 ;; (lookup m d 1 2)
                 (hd/similarity d
                                (torch->jvm
                                 (:result
                                  (lookup m d 1 2))))))
             (binding [*torch-device* :cuda]
               (let [m (sparse-sdm {:address-count 100
                                    :address-density 0.2
                                    :word-length 25})
                     d (hd/->hv)]
                 ;; (decode-address m (hd/->hv) 2)
                 (write m d d 2)
                 (lookup m d 1 2)
                 (hd/similarity d
                                (torch->jvm
                                 (:result
                                  (lookup m d 1 2))))))]]
      (assert r [1.0 1.0]))))





;; -----------------------------------------

(comment
  (do (System/gc) (py.. torch/cuda empty_cache))
  (py/get-item (py.. torch/cuda memory_stats)
               "active.all.current")
  (defn to-gib [bytes] (/ bytes (Math/pow 2 30)))
  (defn to-mib [bytes] (/ bytes (Math/pow 2 20)))
  (to-gib (* (long 1e5) 1))
  (to-gib (py/get-item (py.. torch/cuda memory_stats)
                       "active_bytes.all.current"))
  (to-mib (py/get-item (py.. torch/cuda memory_stats)
                       "active_bytes.all.current")))


;; --------------------------------------------------------------
;;                 +++ Sequence Memory +++
;; --------------------------------------------------------------


(defprotocol SequenceMemory
  (write-xs! [this adress-xs decoder-threshold])
  (lookup-xs [this address decoder-threshold]))

;; --------------------------------------------------------------------------------
;; The Problem:
;;
;; Encoding sequences in a 'pointer' memory like here.
;;
;; input: something like a sequence of addresses [a,b,c,d] = `address-xs`.
;;
;;
;; Version 1:
;;
;; Store each element in the seq with the element preceding it as address
;;
;; store(addr: address-xs[k], content: address-xs[k + 1]), for all elements in address-xs
;;
;;
;;
;;
;;      a     +-->   b      +-> c
;;   +-----+  |   +-----+   |
;;   |     |--+   |     |---+    ...
;;   +-----+      +-----+
;;
;;    k = 0  ,     k = 1
;;
;;
;;  Storing element b with address a
;;  The squence is shown to converge, just like address->address sdm [Kanerva 1988, 2]
;;
;;  This is somewhat equivalent to a linked-list, 'pointer chain' I suppose.
;;
;;
;;
;;
;; This a /first-order sequence memory/ and it breaks down the moment we want to store multiple sequences that share elements.
;; For instance
;;
;; [a,b,c,d,e]  xs1
;; [j,k,l,d,f]  xs2
;;
;; With the first-order sequence memory and those 2 sequences stored, we have a 50/50 chance to recognize the continuation of `d` to be `e` or `f`.
;;
;;

;; Let's build a first order sequence memory:

(defn low-confidence? [confidence] (< confidence 0.1))

(defn first-order-sequence-memory
  []
  (let [sdm
          ;; a small sdm for personal pleasure
          (->sdm {:address-count (long 1e5)
                  :address-density 0.0014
                  :word-length (:bsdc-seg/N
                                 hd/default-opts)})
        stop? (comp low-confidence? :confidence)]
    (reify
      SequenceMemory
        (write-xs! [this address-xs decoder-threshold]
          (doseq [[addr content] (partition 2 1 address-xs)]
            (write sdm addr content decoder-threshold)))
        (lookup-xs [this address decoder-threshold]
          (reduce (fn [{:as acc :keys [address result-xs]}
                       _]
                    ;; for stopping, you could add a
                    ;; termination code, or check the
                    ;; confidence of the result
                    (let [next-outcome
                            (lookup sdm address 1 2)]
                      (if (stop? next-outcome)
                        (ensure-reduced acc)
                        {:address (:result next-outcome)
                         :result-xs (conj result-xs
                                          next-outcome)})))
            {:address address
             :result-xs [{:input? true :result address}]}
            ;; taste
            (range))))))

(comment
  (def xs1 (into [] (repeatedly 5 #(hd/->hv))))
  (->> (let [m (first-order-sequence-memory)]
         (write-xs! m xs1 2)
         (lookup-xs m (first xs1) 2))
       :result-xs
       (map :result))
  (def res *1)
  (count res)
  (map #(hd/similarity %1 %2) (map ensure-jvm res) xs1)
  '(1.0 1.0 1.0 1.0 1.0))

(comment
  (def xs1 (into [] (repeatedly 5 #(hd/->hv))))
  (def xs2 (into [] (repeatedly 5 #(hd/->hv))))
  (let [m (first-order-sequence-memory)]
    (write-xs! m xs1 2)
    (write-xs! m xs2 2)
    [(lookup-xs m (first xs1) 2)
     (lookup-xs m (first xs2) 2)])
  (def res *1)
  (defn similarities
    [xs xsout]
    (map #(hd/similarity %1 %2)
      xs
      (map ensure-jvm (map :result (:result-xs xsout)))))
  (let [[xs1-res xs2-res] res]
    ;; (map :result (:result-xs xs2-res))
    [(similarities xs1 xs1-res) (similarities xs2 xs2-res)])
  '[(1.0 1.0 1.0 1.0 1.0)
    (1.0 1.0 1.0 1.0 1.0)])

(comment
  (let [hv (memoize (fn [_] (hd/->hv)))
        xs1 (into [] (map hv [:a :b :c :d]))
        xs2 (into [] (map hv [:f :h :c :e]))]
    (def xs1 xs1)
    (def xs2 xs2))
  (map #(hd/similarity %1 %2) xs1 xs2)
  ;; .. they overlap in the middle
  '(0.0 0.0 0.0 1.0 0.0)
  (def res
    (let [m (first-order-sequence-memory)]
      (write-xs! m xs1 2)
      (write-xs! m xs2 2)
      [(lookup-xs m (first xs1) 2)
       (lookup-xs m (first xs2) 2)]))
  (defn similarities
    [xs xsout]
    (map #(hd/similarity %1 %2)
      xs
      (map ensure-jvm (map :result (:result-xs xsout)))))
  (let [[xs1-res xs2-res] res]
    ;; (map :result (:result-xs xs2-res))
    [(similarities xs1 xs1-res) (similarities xs2 xs2-res)])
  ;; .. the output is 50/50 between the 2 elements we
  ;; said.
  ;;
  ;; [:a   :b  :c  :d]
  ;; [:f   :h  :c  :e]
  ;; [(1.0 1.0 1.0 0.5)
  ;;  (1.0 1.0 1.0 0.5)]
  ;; ... the last element is similar to both e and d
  ;; We cannot differentiate the 2 sequences. The
  ;; moment a single element that is part of multiple
  ;; seqs comes into our path!
)



;; ---------------------
;; Version 2: (just and idea, a mapped seq)
;; ---------------------
;; - Map each seq into it's own domain,
;; - when reading, unmap to content domain
;; - this solves the `d` problem
;;
;;
;; Downside: user needs to keep track of the sequence handles
;;
;;
;;
;;              [a,b,c,d,e,f] = `xs`
;;
;;       write:      |
;;                   |
;;         +---+     |
;;         |hxs|     v
;;         +---+
;;  1.     + sequence-handle
;;           (->hv)
;;
;;
;;  2.    bind all `xs` with `hsx`s -> `mapped-hsx`
;;
;;              | `mapped-hsx`
;;              | write-xs
;;              v
;;
;;               ha      hb
;;             +----+  +----+  +----+  +----+
;;             | hb +->| hc +->|    +->|    |
;;             +----+  +----+  +----+  +----+
;;                       first order chain
;;
;;
;;  -> (ha, hsx)
;;
;; ----------------------------------
;;
;; you need 2 of the the [a, ha, hsx]
;;
;; (unbind ha a) -> hsx
;; (bind a, hsx) -> ha
;; (unbind ha hsx) -> a
;;
;;
;; read (a hsx):
;; bind (a hsx) -> addr
;; iteratively lookup addr
;; for each outcome `d`, (unbind d hsx)
;; -> mapping it back into the 'real/user' domain
;;
;;
;; Problem: if you have 2 sequences that start with the same addr. can't differentiate
;; (you can't do that with k-fold either though).
;;

;; -----------------------


;; Version 3:
;;
;; K fold memory
;;
;; - In [2] Kanerva is building up the principle with a vocubulary, it is reproduced here, not completely but mostly
;;   and with somewhat different notation that fits the context here:
;;
;; (T is the dataset, like everything the machine encounters).
;;
;; - kth-Order Transition: 'A sequence occuring in xs(T) is a busbsequence of xt(T) of length k + 1
;;   For example [a,b,c,d] is a third-order transition; it says that the three-element seqqquence [a,b,c] is followed
;;   by d.'
;;
;; - j-Step Transition:
;;   A pair of elements in xs(T) separated by j - 1 elements. For example [a,d] is a three-step transition,
;;   it says that a is followed by d.
;;   A first-order transition (here, the 'pointer chain' Version 1) is a one-step transition.
;;   We already see that j-step transitions are easy to store in random access memory.
;;   Store `d` with `a` as address.
;;
;; - j-Step Memory:
;;   A (sparse distributed) memory sdm-j[j] that stores the j-step transitions of xs(T).
;;   The data retrieved from reading with `x` can be denoted data-j[j,x],
;;   or `(lookup-xs sdm-j[j] x)`
;;   It is further assumed that the data become avalable j time steps after the address x has been presented to the memory,
;;   so that the memory has a built-in delay. 👈
;;
;; - k-fold Memory:
;;   A k-fold memory is a set of k j-step memores {sdm-j[1], sdm-j[2], ..., sdm-j[k]}.
;;   The jth memory, sdm-j[j] is also referred to as the jth fold.
;;
;; - k-fold Data:
;;   The k-fold data at time T, D(T), are the multiset of the datta available at time T.
;;
;; - k-fold prediction:
;;   To summarize (vaguely):
;;
;;   Consider having read `a` from a one-fold memory:
;;
;;   [ a, ? ],
;;
;;   the data at hand will be `b'`, and the prediction input came solely from the addresses read with `a` at time t - 1.
;;
;;   In a k-fold prediction, reading at
;;
;;   [a,b,c,d,?],
;;
;;   The addresses read t - 1, t - 2, t - 3, t - 4 all contribute to the prediction of `?`
;;   If T includes sequences:
;;
;;   [a,b,c,d,e]  xs1
;;   [j,k,l,d,f]  xs2
;;
;;   Then, the k-prediction will have roughly
;;
;;   1/k contributions from the xs2 for `f`.
;;
;;   1/k * 4 contributions from xs1 for `e`,
;;
;;   The data will include overwhelmingly more of `e`, than `f`, so `e` is the 4-fold prediction for [a,b,c,d,?]
;;

;;
;; ---------------------------------
;; The delayed address decoder:
;; ---------------------------------
;;
;; 1. split all addresses into `k-delay` buckets
;; So that each address has an associated `delay`. (0 < `delay` <= `k-delays`)
;; (this can be done by delaying the input to the address decoder, or its output lines,
;; which would fit the neurophysiology of the parallel fibers - they are slow conducting).
;;
;;
;;
;; decoding:
;;
;; When decoding address `a` at time `t0`,
;; activate address locations `a0` at time `t0`,
;; then `a1` at time `t1` and so forth.
;;
;;
;;
;;          a
;;                       +----------- delay lines     (~ slow conducting parallel fibers?)
;;          |            |
;;          |            |
;;          v            v
;;     +----------+      +-+     +---+                  +---+
;;     | -----    |  ----+-+-----|   |-------------->   |   | a1
;;     |          |      |1|     |   |                  |   |
;;     | -----    |   ---+-+-->  |   | a0               |   |
;;     |          |      |0|     |   |                  |   |
;;     +----------+      | |     |   |                  |   |
;;       address decoder | |     +---+                  +---+
;;        ^              | |       |
;;        |              | |       |
;;        |              +-+       |
;;     addresses       delays      v
;;
;;
;;                             activations
;;                             {a0}                     {a1}  adress-location activation set
;;
;;
;;                             t0                        t1
;;
;;
;; 2. When decoding, the resulting activation set is the union of active locations,
;;    Including the ones activated (1...k) steps in the past.
;;
;;
;;  t0:
;;  decode with `a`
;;
;;         a
;;
;;      +-------+         +---+
;;      |       |         |   |
;;      |    A  +-------> |   |
;;      |       |         |   |
;;      +-^-----+         +---+
;;        |                   ^
;;        |                   |
;;     future state:        { a0 }  activation set (now)
;;     [ a1, a2, a3, ...]
;;
;;
;;     at each times step,
;;     some a-lines will be on in the future
;;
;;
;;
;;
;;  t1:
;;  the a-lines are still on for a while
;;  decode with `b`
;;
;;
;;        b               +---+
;;        |               |   |
;;        v               |   |
;;      [ A ]  ---------> |   |
;;       ^                |   |
;;       |                +---+
;;    future state:          ^
;;   [a2,a3, ...]            |
;;   [b1,b2, ...]         { a1 , b0 }  activation set (now)
;;
;;
;;
;;
;;
;;
;;

;; ---------------------------------------------------------------
;;
;; Note that in neuroanatomy each line would activate subsequent 'segments' of purkinje cells, (but the whole content matrix here)
;;
;; Conceptually, we split each parallel fiber into k-delay buckets.
;; The mapping is not address-location -> parallel fiber anymore, but
;; address-location set -> parallel fiber
;; where each element in the address-location set represents a 'delay activation line' with one of 0,...,k delays
;; With the difference that we don't model purkinje segments.
;;
;; Presumably this simplification does to not distort the biological plausibility, conceptually.
;; You could wonder about things like interleaving sequences
;; Or how does 'backwards in time' work? Is that supported by cerebellum?
;;
;;
(defn ->address-delays
  "Returns an address-delay table.

                +--- 1 bit per row, the assigned delay for the address.
                v
    +------------------+
    |           1      |  <- address 1
    +------------------+
    |                  |     ...  address-count M ~ 1e6
    |                  |
    +------------------+
      0  1, ..  ^    k-delays (~ 6)
                |
                |
              delay k


   The delay for address `j` is signaled by the on-bit location in the jth address row.

  "
  [{:keys [address-count k-delays]}]
  (let [t (torch/zeros [address-count k-delays]
                       :device *torch-device*
                       :dtype torch/bool)
        idxs [(torch/arange (py.. t (size 0)))
              (torch/randint :low 0
                             :high (py.. t (size 1))
                             :size [(py.. t (size 0))])]]
    (py/set-item! t idxs 1)))

(defn delay-activation-table
  "Returns a tensor of size (address-count, k-delays).


  +---+---+-----------+  address-count
  |   |   |         0 |
  |   |   |         1 |
  |   |   |         0 |
  |   |   |         1 |
  +---+---+-----------+

    t0, t1, ...      k-delays


  remembering delays for address a is done by incrementing the counters for each delay column

                                          delay activation table
  +---+  +----+    +-------------------+
  | 1 |  | 2  |----+------------->1    |
  | 0 |  |    |    |                   |
  | 1 |  | 0  |----+->1                |
  | 0 |  |    |    |                   |
  | 0 |  |    |    |                   |
  +---+  +----+    +-------------------+
   a     delays      t0           t2



  Reading addresses at time t + k means selecting collumn (mod t + k k-delays) from address count.

  Moving time forward is rotating the table, so the first is dropped, and zeroing out the fresh last column,
  then fresh.

  "
  [{:keys [address-count k-delays]}]
  (torch/zeros [address-count k-delays]
               :device *torch-device*
               :dtype torch/bool))

(defn read-delay-table
  "Reads the delay table at a given time step."
  ([delay-table] (read-delay-table delay-table 0))
  ([delay-table future-k]
   (torch/select delay-table
                 :dim 1
                 :index (mod future-k
                             (py.. delay-table (size 0))))))

(comment
  (def t (torch/randn [5 3]))
  (read-delay-table t)
  (read-delay-table t 10))

(defn write-delay-table!
  "Writes the delay information into the delay activation table."
  [delay-table address-locations delays]
  (py/set-item!
   delay-table
   [address-locations]
   (py.. (py/get-item delay-table [address-locations])
     (bitwise_xor (py/get-item delays
                               [address-locations])))))


;; -----------
;; unit tests
;; -----------

(comment
  (let [delays (torch/tensor [[false false true]
                            [true false false]
                            [true false false]])
      address-locations (torch/tensor [false true false])
      dtable (delay-activation-table {:address-count 3
                                      :k-delays 3})]
  (assert
    (torch/equal
      (write-delay-table! dtable address-locations delays)
      (torch/tensor [[false false false] [true false false]
                     [false false false]])))
  (assert (torch/equal (read-delay-table
                         (torch/tensor [[false false false]
                                        [true false false]
                                        [false false
                                         false]]))
                       (torch/tensor [false true false])))
  (assert
    (update-delay-table
      (torch/tensor [[false true false] [true false false]
                     [false false false]]))
    (torch/tensor [[true false false] [false false false]
                   [false false false]]))
  (let [t (torch/randn [3 5])
        ;; reference impl
        update-delay-table-1
          (fn
            ([delay-table]
             (update-delay-table delay-table 1))
            ([delay-table k-steps]
             (reduce (fn [t _] (update-delay-table t))
               delay-table
               (range k-steps))))]
    (assert (torch/equal (update-delay-table t 2)
                         (update-delay-table-1 t 2))))))


(defn update-delay-table
  "Moves time forward by rotating the delay table and zeroing out the new column.

  `k-steps`: How often to move, this is ~ to saying how
  much time steps pass.

  Doesn't support negative time.
  "
  ([delay-table] (update-delay-table delay-table 1))
  ([delay-table k-steps]
   (py..
     (torch/roll delay-table (- k-steps))
     (index_fill_
       1
       (torch/arange (- k-steps) 0 :device *torch-device*)
       false))))

;; -----------------------
;; Delayed Address Decoder
;; -----------------------

(defn ->delayed-address-state
  [{:keys [address-count k-delays]}]
  {:t 0
   ;; the future addresses
   :delay-table (delay-activation-table
                  {:address-count address-count
                   :k-delays k-delays})
   :address-count address-count
   :k-delays k-delays
   :address-delays (->address-delays {:address-count
                                        address-count
                                      :k-delays k-delays})})

(defn read-address-locations
  [{:keys [delay-table]}]
  (read-delay-table delay-table))

(defn move-address-locations
  ([state] (move-address-locations state 1))
  ([{:as state :keys [delay-table t]} t-steps]
   (let [next-table (update-delay-table delay-table
                                        t-steps)]
     (assoc state
       :delay-table next-table
       :t (+ t t-steps)))))

(defn update-future-addresses!
  [state address-locations]
  (let [{:keys [delay-table address-delays]} state]
    (write-delay-table! delay-table
                        address-locations
                        address-delays)))

(defn reset-delay-state
  [state]
  (assoc
   state
   :t 0
   :delay-table (delay-activation-table state)))

;; ----------------------------------

(defprotocol Resets
  (reset [this]))

(defprotocol MovesTime
    (move-time
      [this]
      [this t]))

(defn diverging? [{:keys [result-xs]} result]
  (when-let [last (:confidence (peek result-xs))]
    (< (:confidence result) last)))

;;
;; Just explorative version.
;;
(defn ->k-fold-memory
  [{:as opts
    :keys [k-delays sdm-opts stop? decoder-threshold]}]
  ;;
  ;; Note, I do not model each j-fold with a single
  ;; sdm. I model a single sdm, the address decoder
  ;; delay provides the k-foldedness
  ;;
  (let [decoder (->decoder-coo opts)
        sdm-storage (->sdm-storage-coo opts)
        addr-delay-state (atom (->delayed-address-state opts))
        decode-and-step!
        (fn [addr decoder-threshold]
          (let [address-locations-complete
                ;; 'a'
                ;;
                (decode-address decoder
                                addr
                                decoder-threshold)
                ;; now we split the addresses of 'a'
                ;; into it's delay lines
                _ (update-future-addresses!
                   @addr-delay-state
                   address-locations-complete)
                ;; the address locations of 'now'
                ;; {a0}
                address-locations (read-address-locations @addr-delay-state)
                ;;
                ;; move time forward, so the next
                ;; addresses will contain
                ;; {a1}
                _ (swap! addr-delay-state move-address-locations)]
            address-locations))]
    (reify
      clojure.lang.IDeref
      (deref [this]
        {:addr-delay-state addr-delay-state
         :decoder decoder
         :decode-and-step! decode-and-step!
         :sdm-storage sdm-storage})
      MovesTime
      (move-time [this] (move-time this 1))
      (move-time [this t]
        (if (< t 0)
          (throw
           (Exception.
            "doesn't support moving back in time currently"))
          (swap! addr-delay-state move-address-locations
                 t)))
      Resets
      (reset [this]
        (swap! addr-delay-state reset-delay-state))
      SequenceMemory
      (write-xs! [this address-xs decoder-threshold]
        (doseq [[addr content] (partition 2 1 address-xs)]
          (write-1 sdm-storage
                   (decode-and-step! addr
                                     decoder-threshold)
                   content)))
      (lookup-xs [this address decoder-threshold]
        (reduce (fn [{:as acc :keys [address result-xs]}
                     _]
                  (let [address-set (decode-and-step!
                                     address
                                     decoder-threshold)
                        next-outcome (lookup-1 sdm-storage
                                               address-set
                                               1)]
                    (if-let [stop-reason
                             (stop? acc next-outcome)]
                      (ensure-reduced
                       (assoc acc
                              :stop-reason stop-reason
                              :stop-result next-outcome))
                      {:address (:result next-outcome)
                       :result-xs (conj result-xs
                                        next-outcome)})))
                {:address address
                 :result-xs [{:input? true :result address}]}
                ;; taste
                (range 7))))))


(comment

  (def k-fold-memory
    (->k-fold-memory
     {:k-delays 3
      :stop? (fn [acc next-outcome]
               (when (< (:confidence next-outcome) 0.05)
                 :low-confidence))
      :address-count (long 1e5)
      ;; you need more density, because
      ;; we thin it out over the delays
      :address-density 0.0026
      :word-length (long 1e4)}))

  (let [hv (memoize (fn [_] (hd/->hv)))
        xs1 (into [] (map hv [:a :b :c :d]))
        xs2 (into [] (map hv [:f :h :c :e]))]
    (def xs1 xs1)
    (def xs2 xs2))

  (write-xs! k-fold-memory xs1 2)
  (reset k-fold-memory)
  (def res (lookup-xs k-fold-memory (first xs1) 2))

  (defn similarities [xs xsout]
    (map
     #(hd/similarity %1 %2)
     xs
     (map ensure-jvm (map :result (:result-xs xsout)))))

  (similarities xs1 res)
  '(1.0 1.0 1.0 1.0)

  (do (reset k-fold-memory)
      (write-xs! k-fold-memory xs2 2)
      (def res2
        (lookup-xs k-fold-memory (first xs2) 2)))

  (similarities xs1 res2)
  (0.0 0.0 1.0 0.0)
  (similarities xs2 res2)
  (1.0 1.0 1.0 1.0)

  ;; ... success.

  ;; capacity?

  (def T (into [] (repeatedly 1e3 #(hd/->hv))))
  (def t-sequences
    (into
     []
     (for [n (range 1000)]
       (into
        []
        (for
            [i (range 6)]
            (fm.rand/irand (count T)))))))


  ;; 1.9s for 10 with density 0.002, M = 1e5, depends heavily on the address density
  )


;; ---------------------------------------
;;  Sanity checks
;; ---------------------------------------

(comment
  (def k-fold-memory
    (->k-fold-memory
     {:address-count (long 1e5)
      :address-density 0.00003
      :k-delays 6
      :stop? (fn [acc next-outcome]
               (or ;; (when (diverging?
                ;; acc
                ;; next-outcome)
                ;;   :diverging)
                (when (< (:confidence next-outcome)
                         ;; 0.1 is a
                         ;; good value
                         ;; for 100
                         ;; segment
                         ;; hdvs
                         0.05)
                  :low-confidence)))
      :word-length (long 1e4)}))
  (def a (hd/->seed))
  (def nothing (hd/->empty))
  (let [_ (reset k-fold-memory)
        a0 (-> ((@k-fold-memory :decode-and-step!) a 1)
               (py.. (to "cpu" :dtype torch/int8))
               (ensure-jvm))
        _ (reset k-fold-memory)
        addr-nothing
        (-> ((@k-fold-memory :decode-and-step!) nothing 1)
            (py.. (to "cpu" :dtype torch/int8))
            (ensure-jvm))]
    (assert (= (f/sum addr-nothing) 0.0))
    ;; you can query with nothing, that is the same as
    ;; moving time just saying
    (let [_ (reset k-fold-memory)
          ;; 0
          a0-prime
          (-> ((@k-fold-memory :decode-and-step!) a 1)
              (py.. (to "cpu" :dtype torch/int8))
              (ensure-jvm))
          _ (assert (= a0 a0-prime))
          ;; 1
          _ (move-time k-fold-memory 1)
          ;; 2
          a2-moved-time
          (-> ((@k-fold-memory :decode-and-step!) a 1)
              (py.. (to "cpu" :dtype torch/int8))
              (ensure-jvm))
          _ (reset k-fold-memory)
          ;; 0
          _ (-> ((@k-fold-memory :decode-and-step!) a 1)
                (py.. (to "cpu" :dtype torch/int8))
                (ensure-jvm))
          ;; 1
          _ (->
             ((@k-fold-memory :decode-and-step!) nothing 1)
             (py.. (to "cpu" :dtype torch/int8))
             (ensure-jvm))
          ;; 2
          a2-queried-with-noting
          (-> ((@k-fold-memory :decode-and-step!) a 1)
              (py.. (to "cpu" :dtype torch/int8))
              (ensure-jvm))]
      (assert (= a2-queried-with-noting a2-moved-time)))))


(comment
  (def T (into [] (repeatedly 1e3 #(hd/->hv))))
  (binding [*torch-device* :cuda]
    (do (def k-fold-memory
          (->k-fold-memory
            ;; note the tiny sdm.
            {:address-count (long 1e5)
             :address-density 0.00003
             :k-delays 6
             :stop? (fn [acc next-outcome]
                      (when (< (:confidence next-outcome) 0.05)
                        :low-confidence))
             :word-length (long 1e4)}))
        (time (doseq [t-seq (take 1000 t-sequences)]
                (reset k-fold-memory)
                (write-xs! k-fold-memory (map T t-seq) 1)))
        (def outcome
          (time (doall (for [t-seq (take 1000 t-sequences)]
                         (do (reset k-fold-memory)
                             (similarities
                               (map T t-seq)
                               (lookup-xs k-fold-memory
                                          (T (first t-seq))
                                          1)))))))))
  ;; filter where last element is not 1.0 similar to
  ;; the input
  '(0.05 0.0 0.05 0.05 0.05)
  ;; 5 out of 1000, not bad
  (float (- 1 (/ 5 1000)))
  0.995
  ;; to get beyond 0.99 or something with this kind of
  ;; stuff is hard I think.
)


(comment

  ;; ---------------------------------------------------------
  ;; what is the meaning of top-k > 1 ?
  ;; ---------------------------------------------------------

  (do
    (alter-var-root
     #'hd/default-opts
     #(merge %
             (let [dimensions (long 1e4)
                   segment-count 20]
               {:bsdc-seg/N dimensions
                :bsdc-seg/segment-count segment-count
                :bsdc-seg/segment-length
                (/ dimensions segment-count)})))

    (def m (sparse-sdm {:address-count (long 1e5)
                        :address-density 0.00003
                        :word-length (long 1e4)}))

    (def d (hd/->seed))
    (def d-related (hd/thin (hd/superposition (hd/->seed) d)))

    (do
      (write m d d 1)
      (write m d-related d-related 1))


    ;; top-k = 1 I get something out that is ca 50% d
    (hd/similarity d-related
                   (torch->jvm (:result (lookup m d 1 1))))

    ;; and 100% d
    (hd/similarity d
                   (torch->jvm (:result (lookup m d 1 1))))
    ;; with top-k = 2 you get both of them out,
    ;; querying with with either of them
    ;;
    ;; intuitively, this is because they
    ;; share 50% of their address locations. So
    ;; those bit counters are high, but not the
    ;; highest. with top-k > 1, we *unmask* the
    ;; relatively high
    ;; ("secondary") bit counters, this returns us
    ;; the superposition of *related* addresses
    ;;
    (hd/similarity d
                   (torch->jvm (:result (lookup m d 2 1))))
    (hd/similarity d-related
                   (torch->jvm (:result (lookup m d 2 1))))
    (hd/similarity d-related
                   (torch->jvm
                    (:result (lookup m d-related 2 1))))
    ;; to sanity check, a random seed (not stored)
    ;; will not result in anything resembling
    ;; anything else, even if top-k is 'high'
    (let [e (hd/->seed)]
      (hd/similarity e
                     (torch->jvm (:result
                                  (lookup m e 10 1)))))
    ;; but top-k == segment-length actually results
    ;; in 'everything'
    (= (let [e (hd/->seed)]
         (hd/similarity e
                        (torch->jvm
                         (:result (lookup m e 500 1)))))
       1.0)
    (= (let [e (hd/->seed)]
         (dtt/->tensor (torch->jvm (:result
                                    (lookup m e 500 1)))
                       :datatype
                       :int8))
       (hd/->ones)))


  [0.55 1.0 1.0 1.0 1.0 0.05 true true]

  (do
    (alter-var-root
     #'hd/default-opts
     #(merge %
             (let [dimensions (long 1e4)
                   segment-count 20]
               {:bsdc-seg/N dimensions
                :bsdc-seg/segment-count segment-count
                :bsdc-seg/segment-length
                (/ dimensions segment-count)})))

    (def m (sparse-sdm {:address-count (long 1e5)
                        :address-density 0.00003
                        :word-length (long 1e4)}))

    (def d (hd/->seed))
    (def d-related (hd/thin (hd/superposition (hd/->seed) d)))

    ;; heteroassociative:
    ;; d->a
    ;; d-related->b

    (def a (hd/->seed))
    (def b (hd/->seed))

    (do
      (write m d a 1)
      (write m d-related b 1))

    ;; sanity check, the address content should be different from the address
    (hd/similarity d
                   (torch->jvm (:result (lookup m d 1 1))))
    0.0

    ;; query with
    ;; top-k = 1

    [:a
     (hd/similarity a (torch->jvm (:result (lookup m d 1 1))))
     :b
     (hd/similarity b (torch->jvm (:result (lookup m d 1 1))))]
    [:a 1.0 :b 0.0]

    ;; top-k = 2
    [:a
     (hd/similarity a (torch->jvm (:result (lookup m d 2 1))))
     :b
     (hd/similarity b (torch->jvm (:result (lookup m d 2 1))))]
    [:a 1.0 :b 1.0]

    ;; ... look I get the superposition of both a and b,
    ;; even though a and b are unrelated, they are possible continuations of `d`
    ;; but `b` is only unmasked secondarily, in a more 'divergent mode' of querying.
    ))



;; -------------------------------------

(defn k-fold-sdm
  [opts]
  (let [inner-decoder (->decoder-coo opts)
        addr-delay-state (atom (->delayed-address-state
                                 opts))
        decode-and-step!
          (fn [addr decoder-threshold]
            (let [address-locations-complete
                    ;; 'a'
                    ;;
                    (decode-address inner-decoder
                                    addr
                                    decoder-threshold)
                  ;; now we split the addresses of 'a'
                  ;; into it's delay lines
                  _ (update-future-addresses!
                      @addr-delay-state
                      address-locations-complete)
                  ;; the address locations of 'now'
                  ;; {a0}
                  address-locations (read-address-locations
                                      @addr-delay-state)
                  ;;
                  ;; move time forward, so the next
                  ;; addresses will contain
                  ;; {a1}
                  _ (swap! addr-delay-state
                      move-address-locations)]
              address-locations))
        storage (->sdm-storage-coo opts)]
    (reify
      MovesTime
        (move-time [this] (move-time this 1))
        (move-time [this t]
          (if (< t 0)
            (throw
              (Exception.
                "doesn't support moving back in time currently"))
            (swap! addr-delay-state move-address-locations
              t)))
      Resets
        (reset [this]
          (swap! addr-delay-state reset-delay-state))
      SDM
        (write [this address content decoder-threshold]
          (write-1 storage
                   (decode-and-step! address
                                     decoder-threshold)
                   content))
      (lookup
          [this address top-k decoder-threshold]
          (lookup-1 storage
                    (decode-and-step! address
                                      decoder-threshold)
                    top-k)))))







;; -------------------------------------






;; ------------------
;; Literature:
;; ------------------
;;
;; 1
;; Pentti Kanerva 1993 Sparse Distributed Memory and Related Models
;;
;; 2
;; Pentti Kanerva /Sparse Distributed Memory/, 1988
;;
;; 3
;; Jaeckel, L.A. 1989a. An Alternative Design for a Sparse Distributed Memory.
;; Report RIACS TR 89.28, Research Institute for Advanced Computer Science,
;; NASA Ames Research Center.
;;
;; 4
;; Jaeckel, L.A. 1989b. A Class of Designs for a Sparse Distributed Memory. Report
;; RIACS TR 89.30, Research Institute for Advanced Computer Science, NASA
;; Ames Research Center.
;;
;; 5
;; https://nextjournal.com/cdeln/reference-counting-in-clojure
;;
