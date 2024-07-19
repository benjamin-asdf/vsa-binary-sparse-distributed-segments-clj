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
;; And mapping to neurophysiology of cerebellum
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


  ;; I played around with this until I found a combination where this delta was small.
  (-
   (* 100 (ideal-p 1e3 1e4))
   (* 100
      (calculate-activation-probability 1e4
                                        (/ 100 1e4)
                                        0.0009
                                        2
                                        2000000)))


  (-
   (* 100 (ideal-p 1e4 1e6))
   (* 100
      (calculate-activation-probability
       1e4
       0.0014

       (/ 20 1e4)
       ;; 0.000001
       2
       2000000)))



  (calculate-activation-probability 1e4 (/ 100 1e4) 0.0009 2 2000000)
  0.003801
  (calculate-activation-probability 1e4 (/ 100 1e4) 0.0009 2 2000000)
  ;; 0.003872

  ;; for
  ;; T = 1.000
  ;; M = 10.000
  ;; address density = 0.0009, decoder threshold = 2
  ;; seem to be in a good balpark
  ;; (unless the calculation is wrong)


  (calculate-activation-probability
   1e4
   (/ 20 1e4)
   0.005
   2
   2000000))


;; ----------------------
;; Concrete torch implementation
;; ----------------------

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
  ;; https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815
  ;; float8 is experimental as far as I understand.
  ;; they don't have a matmul implementation for
  ;; integer types. using float16 here for now. If you
  ;; wanted and you run on cpu, you could go to uint8
  (py.. (torch/less (torch/rand [address-count
                                 address-length]
                                :device
                                *torch-device*)
                    density)
    (to torch/float16)))


;; kinda funny that many rows are never activated
;; in lue of the amount of granule cells, you can wonder
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


;; (torch/add
;;  (torch/tensor [1 2 3] :dtype torch/uint8 :device *torch-device*)
;;  (torch/tensor [1 255 40] :dtype torch/uint8 :device *torch-device*))


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
  ;; d
  (let [address (pyutils/ensure-torch address)
        inputs (torch/mv address-matrix address)
        activations (torch/ge inputs
                              (torch/tensor
                                decoder-threshold
                                :device *torch-device*
                                :dtype torch/float32))]
    ;; y
    activations))

(comment
  (decode-addresses
   (->address-matrix 3 3 0)
   (torch/tensor [0 1 1] :dtype torch/float16 :device *torch-device*)
   1)

  (decode-addresses-coo
   (->address-matrix-coo 3 3 0)
   (torch/tensor [0 1 1] :dtype torch/float32 :device *torch-device*)
   1))

(defn write!
  [content-matrix address-locations input-word]
  (let [input-word (pyutils/ensure-torch input-word
                                        )]
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
      ;; (py.. torch/cuda synchronize)
      ;; (py-ffi/Py_DecRef update)
      ;; (py-ffi/Py_DecRef values)
      ;; (py-ffi/Py_DecRef indices)
      content-matrix)))





(comment



  (binding [*torch-device* :cpu]
    (let [addr (torch/tensor [1 2 4]
                             :dtype torch/long
                             :device *torch-device*)
          word (torch/tensor [1 1 0 0 1] :dtype torch/float16)
          C (torch/sparse_coo_tensor :size [5 5]
                                     :device *torch-device*
                                     :dtype torch/float32)
          counter-max 3]
      (let [activated-locations (py.. (torch/nonzero addr)
                                  (view -1))
            word-nonzero (py.. (torch/nonzero word) (view -1))
            indices (py.. (torch/cartesian_prod
                           activated-locations
                           word-nonzero)
                      (t))
            values (torch/ones (py.. indices (size 1)))
            update (torch/sparse_coo_tensor indices
                                            values
                                            (py.. C size))
            C
            ;; (to_dense)
            (py.. C (add_ update) (coalesce))]
        (py.. C values (clamp_ 0 counter-max))
        (py.. C (to_dense)))))




  (do (alter-var-root
       #'hd/default-opts
       (constantly (let [dimensions 25
                         segment-count 5]
                     {:bsdc-seg/N dimensions
                      :bsdc-seg/segment-count segment-count
                      :bsdc-seg/segment-length
                      (/ dimensions segment-count)})))
      (binding [*torch-device* :cpu]
        (let [word (hd/->hv)
              address-matrix (->address-matrix-coo 100 25 0.2)
              content-matrix (->content-matrix-coo 100 25)]
          ;; (decode-addresses-coo address-matrix word 2)
          content-matrix
          ;; (py.. content-matrix (to_dense))
          ;; (hd/similarity
          ;;  word
          ;;  (torch->jvm
          ;;   (:result
          ;;    (sdm-read-coo content-matrix
          ;;                  (decode-addresses-coo
          ;;                  address-matrix word 2)
          ;;                  1))))
          (write-coo!
           content-matrix
           (decode-addresses-coo address-matrix word 2)
           word)))))

(defn read-coo-1
  [content-matrix address-locations]
  (let [addr-indices (torch/squeeze (torch/nonzero
                                      address-locations)
                                    1)
        sum (py.. (torch/sum (torch/index_select
                               content-matrix
                               0
                               addr-indices)
                             0)
                  (to_dense))]
    (py-ffi/Py_DecRef addr-indices)
    sum))

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


(defn sdm-read-coo
  "Returns a result and a confidence value.

  `result`: A binary segmented hypervector reading from `address-locations`.
  `confidence`: The normalized sum of storage counters that contribute to the result.

  This repesents an approximation of the confidence of the result being an item in memory.
  If `top-k` > 1 this would increase accordingly.
  This can exceed 1, if counter locations are higher than 1, then an item was stored multiple times.

  If close to one, the confidence is high. If close to zero, the confidence is low.

  `top-k`: number of non zero bits to take from each segment.

  If `top-k` == 1, the output is maximally sparse.
  "
  ([content-matrix address-locations top-k]
   (sdm-read-coo content-matrix
                 address-locations
                 top-k
                 hd/default-opts))
  ([content-matrix address-locations top-k
    {:bsdc-seg/keys [segment-count segment-length N]}]
   (py/with-gil
     (let [s (read-coo-1 content-matrix address-locations)
           topk-result (-> s
                           (torch/reshape [segment-count
                                           segment-length])
                           (torch/topk
                             (min top-k segment-length)))
           result (torch/scatter (torch/zeros
                                   [segment-count
                                    segment-length]
                                   :dtype torch/float32
                                   :device *torch-device*)
                                 1
                                 (py.. topk-result -indices)
                                 1)
           address-location-count
             (py.. (torch/sum address-locations) item)]
       ;; (py-ffi/Py_DecRef s)
       ;; (py-ffi/Py_DecRef result)
       ;; (py-ffi/Py_DecRef topk-result)
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
                              ;; taste. It turns out to
                              ;; say high confidence,
                              ;; if the count of
                              ;; address is low, when
                              ;; the address is a
                              ;; 'weak' (sub sparsity)
                              ;; hypervector. The
                              ;; caller would have to
                              ;; take this into account
                              ;; themselves.
                              ;;
                              (* segment-count
                                 address-location-count))
                            item))
        :result (torch/reshape result [N])}))))

(defn sdm-read
  "Returns a result and a confidence value.

  `result`: A binary segmented hypervector reading from `address-locations`.
  `confidence`: The normalized sum of storage counters that contribute to the result.

  This repesents an approximation of the confidence of the result being an item in memory.
  If `top-k` > 1 this would increase accordingly.
  This can exceed 1, if counter locations are higher than 1, then an item was stored multiple times.

  If close to one, the confidence is high. If close to zero, the confidence is low.

  `top-k`: number of non zero bits to take from each segment.

  If `top-k` == 1, the output is maximally sparse.
  "
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
                  ;; also divide by top-k?
                  ;;
                  ;; scaling this with the
                  ;; address-locations
                  ;; count is probably
                  ;; taste. It turns out to
                  ;; say high confidence,
                  ;; if the count of
                  ;; address is low, when
                  ;; the address is a
                  ;; 'weak' (sub sparsity)
                  ;; hypervector. The
                  ;; caller would have to
                  ;; take this into account
                  ;; themselves.
                  ;;
                  (* segment-count address-location-count))
                item))
      :result (torch/reshape result [N])})))



;; not sure yet

(defprotocol SDM
  (known?
    [this address]
    [this address decoder-threshold])
  (lookup-1
    [this address-locations top-k])
  (lookup
    [this address top-k]
    [this address top-k decoder-threshold])
  (converged-lookup
    [this address top-k]
    [this address top-k decoder-threshold])
  (write
    [this address content]
    [this address content decoder-threshold])
  (write-1
    [this address-locations content])
  (decode-address
    [this address decoder-threshold]))

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
      SDM
      (decode-address [this address decoder-threshold]
        (decode-addresses address-matrix
                          address
                          decoder-threshold))
      (write-1 [this address-locations content]
        (write! content-matrix address-locations content))
      (write [this address content decoder-threshold]
        (write-1
         this
         (decode-address this address decoder-threshold)
         content))
      (lookup-1 [this address-locations top-k]
        (sdm-read content-matrix address-locations top-k))
      (lookup [this address top-k decoder-threshold]
        (lookup-1
         this
         (decode-address this address decoder-threshold)
         top-k)))))

(defn sparse-sdm
  [{:keys [address-count word-length address-density]}]
  (let [content-matrix (->content-matrix-coo address-count
                                             word-length)
        address-matrix (->address-matrix-coo
                         address-count
                         word-length
                         address-density)]
    (reify
      SDM
        (decode-address [this address decoder-threshold]
          (decode-addresses-coo address-matrix
                                address
                                decoder-threshold))
        (write-1 [this address-locations content]
          (write-coo! content-matrix
                      address-locations
                      content))
        (write [this address content decoder-threshold]
          (write-1
            this
            (decode-address this address decoder-threshold)
            content))
        (lookup-1 [this address-locations top-k]
          (sdm-read-coo content-matrix
                        address-locations
                        top-k))
        (lookup [this address top-k decoder-threshold]
          (lookup-1
            this
            (decode-address this address decoder-threshold)
            top-k)))))

(comment





  (binding [*torch-device* :cpu]
    (do (alter-var-root
         #'hd/default-opts
         (constantly (let [dimensions 25
                           segment-count 5]
                       {:bsdc-seg/N dimensions
                        :bsdc-seg/segment-count segment-count
                        :bsdc-seg/segment-length
                        (/ dimensions segment-count)})))
        (let [m (dense-sdm {:address-count 100
                            :address-density 0.2
                            :word-length 25})
              d (hd/->hv)]
          (decode-address m (hd/->hv) 2)
          (write m d d 2)
          ;; (lookup m d 1 2)
          ;; (hd/similarity d
          ;;                (torch->jvm
          ;;                 (:result (lookup m d 1 2))))
          )))







  (binding [*torch-device* :cpu]
    (do (alter-var-root
         #'hd/default-opts
         (constantly (let [dimensions 25
                           segment-count 5]
                       {:bsdc-seg/N dimensions
                        :bsdc-seg/segment-count
                        segment-count
                        :bsdc-seg/segment-length
                        (/ dimensions segment-count)})))
        (let [m (sparse-sdm {:address-count 100
                             :address-density 0.2
                             :word-length 25})
              d (hd/->hv)]
          (decode-address m (hd/->hv) 2)
          (write m d d 2)
          ;; (lookup m d 1 2)
          ;; (hd/similarity d
          ;;                (torch->jvm
          ;;                 (:result (lookup m d 1 2))))
          )))







  [
   (binding [*torch-device* :cpu]
     (do (alter-var-root
          #'hd/default-opts
          (constantly (let [dimensions 25
                            segment-count 5]
                        {:bsdc-seg/N dimensions
                         :bsdc-seg/segment-count
                         segment-count
                         :bsdc-seg/segment-length
                         (/ dimensions segment-count)})))
         (let [m (sparse-sdm {:address-count 100
                              :address-density 0.2
                              :word-length 25})
               d (hd/->hv)]
           (decode-address m (hd/->hv) 2)
           (write m d d 2)
           (lookup m d 1 2)
           (hd/similarity d
                          (torch->jvm
                           (:result (lookup m d 1 2)))))))]


  [1.0 1.0])




(comment


  (binding [*torch-device* :cpu]
    (do (alter-var-root
         #'hd/default-opts
         (constantly (let [dimensions 25
                           segment-count 5]
                       {:bsdc-seg/N dimensions
                        :bsdc-seg/segment-count
                        segment-count
                        :bsdc-seg/segment-length
                        (/ dimensions segment-count)})))
        (let [m (sparse-sdm {:address-count 100
                             :address-density 0.2
                             :word-length 25})
              d (hd/->hv)]
          (decode-address m (hd/->hv) 2)
          (write m d d 2)
          ;; (lookup m d 1 2)
          ;; (hd/similarity d
          ;;                (torch->jvm
          ;;                 (:result (lookup m d 1 2))))
          )))




  (binding [*torch-device* :cuda]
    (do (alter-var-root
         #'hd/default-opts
         (constantly (let [dimensions 25
                           segment-count 5]
                       {:bsdc-seg/N dimensions
                        :bsdc-seg/segment-count segment-count
                        :bsdc-seg/segment-length
                        (/ dimensions segment-count)})))
        (let [m (sparse-sdm {:address-count (long 1e6)
                             :address-density 0.002
                             :word-length 25})
              d (hd/->hv)]
          ;; (torch/sum (decode-address m (hd/->hv) 2))
          (write m d d 2)
          ;; (lookup m d 1 2)
          (hd/similarity d
                         (torch->jvm (:result
                                      (lookup m d 1 2)))))))



  ;; M = 1e6, N = 1e4, addr-density = 0.0005
  ;; Kinda cool that I can do it with 1 million locations
  ;; in some ways mirroring the efficiency of the brain sparse endoding
  ;; (only that here its storage and in brain its energy)


  (def T (into [] (map #(pyutils/ensure-torch % :cuda) (repeatedly 1e3 #(hd/->hv)))))

  ;; The memory requirements for the coo matrix is roughly

  (defn c-requirement
    [address-locations-per-input T bit-per-input]
    (let [nse (* address-locations-per-input T bit-per-input)
          ndim 2
          itemsize (py.. torch/uint8 -itemsize)]
      (* nse (+ (* ndim 8) itemsize))))

  (to-mib (c-requirement 40 (long 1e3) 20))


  (time
   (binding [*torch-device* :cuda]
     (do (alter-var-root
          #'hd/default-opts
          (constantly (let [dimensions (long 1e4)
                            segment-count 20]
                        {:bsdc-seg/N dimensions
                         :bsdc-seg/segment-count
                         segment-count
                         :bsdc-seg/segment-length
                         (/ dimensions segment-count)})))
         (let [m (sparse-sdm {:address-count (long 1e5)
                              :address-density 0.0003
                              :word-length (long 1e4)})]
           (doseq [t (take 100 T)]
             ;; (decode-address m t 2)
             (write m t t 2)
             )
           (let [d (first T)]
             (hd/similarity (torch->jvm d)
                            (torch->jvm
                             (:result (lookup m d 1 2)))))
           (torch/sum (decode-address m (first T) 2))))))


  ;; M = 1e6


  (time
   (binding [*torch-device* :cuda]
     (do
       (alter-var-root
        #'hd/default-opts
        (constantly (let [dimensions (long 1e4)
                          segment-count 20]
                      {:bsdc-seg/N dimensions
                       :bsdc-seg/segment-count segment-count
                       :bsdc-seg/segment-length
                       (/ dimensions segment-count)})))
       (let [m (sparse-sdm {:address-count (long 1e6)
                            :address-density 0.0001
                            :word-length (long 1e4)})]
         (doseq [[idx t] (map-indexed vector (take 1000 T))]
           ;; lul like this python and torch end up
           ;; cleanup up
           (when (mod idx 100) (System/gc))
           (write m t t 2))
         (torch/sum (decode-address m (first T) 2))
         (let [d (first T)]
           (hd/similarity (torch->jvm d)
                          (torch->jvm
                           (:result (lookup m d 1 2)))))))))


  (to-mib (c-requirement 380 (long 1e3) 20))


  ;; with this config, 10 fell fast

  (time
   (binding [*torch-device* :cuda]
     (do (alter-var-root
          #'hd/default-opts
          (constantly (let [dimensions (long 1e4)
                            segment-count 20]
                        {:bsdc-seg/N dimensions
                         :bsdc-seg/segment-count
                         segment-count
                         :bsdc-seg/segment-length
                         (/ dimensions segment-count)})))
         (let [m (sparse-sdm {:address-count (long 1e6)
                              :address-density 0.0002
                              :word-length (long 1e4)})]
           (doseq
               [[idx t]
                (map-indexed vector (take 10 T))]
             ;; lul like this python and torch end up
             ;; cleanup up
             ;; (when (mod idx 100) (System/gc))
               (write m t t 2))
           (let [d (first T)]
             (hd/similarity (torch->jvm d)
                            (torch->jvm
                             (:result (lookup m d 1 2)))))
           (torch/sum (decode-address m (first T) 2))))))






  ;; ... but 100 ms is too long
  ;; I want to finish each operation in 2ms preferably
  ;; so that one could do 500 operations in a second

  (def T (into [] (map #(pyutils/ensure-torch % :cpu) (repeatedly 1e3 #(hd/->hv)))))












  ;; M = 1e5 works on cpu

  (time
   (binding [*torch-device* :cpu]
     (do
       (alter-var-root
        #'hd/default-opts
        (constantly (let [dimensions (long 1e4)
                          segment-count 20]
                      {:bsdc-seg/N dimensions
                       :bsdc-seg/segment-count
                       segment-count
                       :bsdc-seg/segment-length
                       (/ dimensions segment-count)})))
       (let [m (sparse-sdm {:address-count (long 1e5)
                            :address-density 0.002
                            :word-length (long 1e4)})]
         (doseq [t (take 100 T)]
           ;; (decode-address m t 2)
           (write m t t 2)
           )
         (let [d (first T)]
           (write m d d 2)
           (lookup m d 1 2)
           (hd/similarity (torch->jvm d)
                          (torch->jvm (:result
                                       (lookup m d 1
                                               2)))))
         ;; (torch/sum (decode-address m (first T) 2))
         ))))
  ;; 4s


  )













;; =========================
;; k fold sdm
;; =========================
;;
;; Pentti Kanerva /Sparse Distributed Memory/, 1988
;;
;; - this requires us to add a bookeeping
;; 1. Each hard location has an associated hard delay (k-delay)
;; 2. When activating locations (should be for reading and writing),
;; - decode addresses like usual
;; - read or write at time step t0, using only address t0 locations
;; - in the next timestep address t1 locations are active (in addition to any other)
;;   I.e. the active locations is the union of active locations, which might be the locations from a
;;   j steps in the past.
;; -
;;



;; in k-fold memory, it is sufficient to
;; stochastically allocate 0,1,2,...k-delays to
;; addresses. Since address decoding is stochastic,
;; you get a mix of delayed address locations.
(defn ->address-delays
  "Returns `addresses-count` address delays distributed over `k-delays`.

  "
  [address-count k-delays]
  ;; (torch/randint k-delays [address-count] :dtype
  ;; torch/uint8 :device *torch-device*)
  (dtt/clone (dtt/compute-tensor [address-count]
                                 (fn [_]
                                   (fm.rand/irand k-delays))
                                 :int8)))

(comment
  (->address-delays 10 5))

;; this whole address bookeeping should not be so many since addresses are so sparse...
;; We can do this on jvm and upgrade when needed

(defn ->address-state
  [delay-index k-delays]
  {:delay-index delay-index
   :k-delays k-delays
   :t 0
   ;; keeping track of the future
   :t->activations {}})

(defn with-activations
  [{:as state :keys
    [t delay-index t->activations k-delays]}
   activated-locations]
  (let [activated-locations
          (if-not (dtt/tensor? activated-locations)
            (pyutils/torch->jvm (torch/squeeze
                                  (torch/nonzero
                                    activated-locations)))
            activated-locations)]
    (assoc state
      :t->activations
        (reduce (fn [t->activations [i-activation k-delay]]
                  (update-in t->activations
                             [(+ t k-delay)]
                             (fnil conj #{})
                             i-activation))
          t->activations
          (map vector
            activated-locations
            (dtt/select delay-index activated-locations)))))
  ;; (-> k-fold-active-locations)
  )

(defn indices->address-locations
  [address-count indices]
  (->address-locations
   address-count
   (torch/tensor
    (into [] indices)
    :dtype torch/long
    :device *torch-device*)))

(defn k-fold-active-locations
  [{:keys [t t->activations delay-index]}]
  (indices->address-locations
   (dtype/ecount delay-index)
   (t->activations t)))

(defn k-fold-step
  [{:as state :keys [t k-delays]}]
  (-> state
      (update :t->activations dissoc t)
      ;;
      ;; I guess you have several options here. In this
      ;; case, this address time flow is circular
      ;;
      ;; In another version, I want to keep the history
      ;; around for going the other way etc.
      ;;
      ;;
      (update :t inc)))

;; -----------------------------------------

(defprotocol AddressDecoder
  (decode [this address decoder-threshold]))

(defn ->address-decoder
  "
  Returns an sdm AddressDecoder"
  [{:keys [address-count word-length address-density]}]
  (let [address-matrix (->address-matrix address-count
                                         word-length
                                         address-density)]
    (reify
      AddressDecoder
      (decode [_ address decoder-threshold]
        (decode-addresses address-matrix
                          address
                          decoder-threshold)))))

(defn auto-associate!
  [content-matrix address decoder decoder-threshold]
  (write! content-matrix
          (decode decoder address decoder-threshold)
          address))

(defn lookup-iteratively
  [content-matrix address decoder
   {:as opts
    :keys [decoder-threshold top-k read-threshold]}]
  ;; lookup iteratively, if you are within critical
  ;; distance this will converge to the output word
  ;; within a few steps
  ;; Kanerva 1988 [2]
  ;;
  (reduce (fn [{:keys [last-outcome address]} step]
            (let [address (pyutils/ensure-torch address)
                  outcome (torch->jvm
                           (sdm-read
                            content-matrix
                            (decode
                             decoder
                             address
                             decoder-threshold)
                            top-k
                            read-threshold))]
              (cond (and last-outcome
                         (< 0.98
                            (hd/similarity last-outcome
                                           outcome
                                           opts)))
                      (ensure-reduced {:last-outcome outcome
                                       :result outcome
                                       :step step})
                    :else {:address outcome
                           :last-outcome outcome
                           :step step})))
    {:address address :step 0}
    (range 6)))


(comment
  (do (System/gc) (py.. torch/cuda empty_cache))


  (py/get-item (py.. torch/cuda memory_stats) "active.all.current")


  (defn to-gib [bytes]
    (/ bytes (Math/pow 2 30)))
  (defn to-mib [bytes]
    (/ bytes (Math/pow 2 20)))
  (to-gib (* (long 1e5) 1))



  (to-gib (py/get-item (py.. torch/cuda memory_stats) "active_bytes.all.current"))


  (to-mib (py/get-item (py.. torch/cuda memory_stats) "active_bytes.all.current"))


  )


(comment
  (do
    (do (System/gc) (py.. torch/cuda empty_cache))
    (alter-var-root #'hd/default-opts
                    (constantly
                     (let [dimensions (long 1e4)
                           segment-count 20]
                       {:bsdc-seg/N dimensions
                        :bsdc-seg/segment-count segment-count
                        :bsdc-seg/segment-length
                        (/ dimensions segment-count)})))
    (let [address-count (long 1e4)
          word-length (:bsdc-seg/N hd/default-opts)
          address-density 0.005
          decoder-threshold 2
          state {:content-matrix (->content-matrix
                                  address-count
                                  word-length)
                 :decoder (->address-decoder
                           {:address-count address-count
                            :address-density address-density
                            :word-length word-length})}
          t (hd/->hv)
          t-prime (hd/weaken t 0.5)
          tb (hd/thin (hd/bundle t (hd/->hv)))
          T (repeatedly 1e3 #(hd/->hv))
          ;; if I don't thin, I get the t out
          tc (hd/bundle t (hd/->hv) (hd/->hv) (hd/->hv))
          addresses
          (decode (:decoder state) t decoder-threshold)]
      (doseq [data T]
        (auto-associate! (:content-matrix state)
                         data
                         (:decoder state)
                         decoder-threshold))
      (auto-associate! (:content-matrix state)
                       t
                       (:decoder state)
                       decoder-threshold)
      ;; [(torch/sum addresses)
      ;;  (torch/sum
      ;;   (decode (:decoder state) t-prime
      ;;   decoder-threshold))
      ;;  (torch/sum
      ;;   (decode (:decoder state) tb
      ;;   decoder-threshold))]
      [(let [r (sdm-read (:content-matrix state) addresses 1)]
         [:sim-t-res
          (hd/similarity (torch->jvm (:result r)) t)
          :confidence (:confidence r)])
       ;; prime
       (let [r (sdm-read
                (:content-matrix state)
                (decode (:decoder state)
                        t-prime
                        decoder-threshold)
                1)]
         [:sim-t-prime-res
          (hd/similarity (torch->jvm (:result r)) t)
          :confidence (:confidence r)])
       (let [r (sdm-read (:content-matrix state)
                         (decode (:decoder state)
                                 tb
                                 decoder-threshold)
                         1)]
         [:sim-tb (hd/similarity (torch->jvm (:result r)) t)
          :confidence (:confidence r)])
       (let [r (sdm-read (:content-matrix state)
                         (decode (:decoder state)
                                 tc
                                 decoder-threshold)
                         1)]
         [:sim-tc (hd/similarity (torch->jvm (:result r)) t)
          :confidence (:confidence r)])]))

  [[:sim-t-res 1.0 :confidence 1.0]
   [:sim-t-prime-res 1.0 :confidence 0.9999999403953552]
   ;; intermediate confidence when made from equal parts
   [:sim-tb 1.0 :confidence 0.3636363446712494]
   ;; low confidence, but correct
   [:sim-tc 1.0 :confidence 0.06641285866498947]])



;; --------------------------------------

(defprotocol KFoldAddressDecoder
  (decode-and-step! [this address decoder-threshold])
  (clear-activations [this])
  (get-state [this]))

(defn ->k-fold-address-decoder
  [{:as opts
    :keys [address-count word-length address-density
           k-delays]}]
  (let [decoder (->address-decoder opts)
        delay-index (->address-delays address-count
                                      k-delays)
        state (atom (->address-state delay-index k-delays))]
    (reify
      KFoldAddressDecoder
        (get-state [this] @state)
        (decode-and-step! [this address decoder-threshold]
          (let [new-locations
                  (decode this address decoder-threshold)
                s (with-activations @state new-locations)
                activations (k-fold-active-locations s)]
            (reset! state (k-fold-step s))
            activations))
        (clear-activations [_]
          (reset! state (->address-state delay-index
                                         k-delays)))
      AddressDecoder
        (decode [this address decoder-threshold]
          (decode decoder address decoder-threshold)))))

(comment

  (let [address-count 100
        word-length (:bsdc-seg/N hd/default-opts)
        address-density 0.05
        decoder-threshold 2
        T (repeatedly 100 #(hd/->hv))
        history (atom [])
        decoder (->k-fold-address-decoder
                 {:address-count address-count
                  :address-density address-density
                  :k-delays 5
                  :word-length word-length})

        _  (decode-and-step! decoder (first T) 2)
        s1 (get-state decoder)
        _ (decode-and-step! decoder (first T) 2)
        s2 (get-state decoder)
        _ (decode-and-step! decoder (first T) 2)
        s3 (get-state decoder)]
    [[(count (-> s1 :t->activations (get 2)))
      (count (-> s2 :t->activations (get 2)))
      (count (-> s3 :t->activations (get 2)))
      (clojure.set/intersection
       (-> s1 :t->activations (get 2))
       (-> s2 :t->activations (get 2))
       (-> s3 :t->activations (get 2)))]
     [(count (-> s1 :t->activations (get 3)))
      (count (-> s2 :t->activations (get 3)))
      (count (-> s3 :t->activations (get 3)))
      (clojure.set/intersection
       (-> s1 :t->activations (get 3))
       (-> s2 :t->activations (get 3))
       (-> s3 :t->activations (get 3)))]])
  ;; [[20 37 0 nil] [25 45 62 #{0 24 92 48 75 99 31 91 33 13 41 64 51 3 66 97 68 83 53 38 30 10 80 8 84}]]




  (do
    (do (System/gc) (py.. torch/cuda empty_cache))
    (alter-var-root #'hd/default-opts
                    (constantly
                     (let [dimensions (long 1e4)
                           segment-count 20]
                       {:bsdc-seg/N dimensions
                        :bsdc-seg/segment-count segment-count
                        :bsdc-seg/segment-length
                        (/ dimensions segment-count)})))
    (let [address-count (long 1e3)
          word-length (:bsdc-seg/N hd/default-opts)
          address-density 0.03
          decoder-threshold 2
          state {:content-matrix (->content-matrix
                                  address-count
                                  word-length)
                 :decoder (->k-fold-address-decoder
                           {:address-count address-count
                            :address-density address-density
                            :k-delays 5
                            :word-length word-length})}
          T (repeatedly 1e3 #(hd/->hv))
          history (atom [])]
      (doseq [data (take 5 T)]
        (swap! history conj
               (-> state
                   :decoder
                   get-state))
        (write! (:content-matrix state)
                (decode-and-step! (:decoder state)
                                  data
                                  decoder-threshold)
                data))
      (clear-activations (:decoder state))
      (let [read1 (fn [t]
                    (sdm-read (:content-matrix state)
                              (decode-and-step!
                               (:decoder state)
                               t
                               decoder-threshold)
                              1))
            out1 (read1 (first T))
            out2 (read1 (:result out1))]
        [:t0
         (hd/similarity (torch->jvm (:result out1)) (first T))
         (hd/similarity (torch->jvm (:result out1))
                        (second T)) :t1
         (hd/similarity (torch->jvm (:result out2)) (first T))
         (hd/similarity (torch->jvm (:result out2))
                        (second T))])))


  [:t0 1.0 0.0 :t1 0.0 1.0]


  (defn read-sequence!
    [{:keys [content-matrix decoder]} address
     decoder-threshold]
    ;;
    ;; stop? Perhaps when the confidence is very low? Or
    ;; when you encounter a stop codon? Or after x
    ;; steps? Would be cool to check the confidence
    ;; then perhaps return random noice instead (that
    ;; would sound biological to me)
    ;;
    (reductions (fn [address _]
                  (:result (sdm-read content-matrix
                                     (decode-and-step!
                                      decoder
                                      address
                                      decoder-threshold)
                                     1)))
                address
                (range 5)))

  (defn cleanup
    [T q]
    (ffirst (sort-by second
                     (fn [a b]
                       (compare (hd/similarity b q)
                                (hd/similarity a q)))
                     (into [] T))))



  (do
    (do (System/gc)
        (py.. torch/cuda empty_cache))
    (alter-var-root
     #'hd/default-opts
     (constantly (let [dimensions (long 1e4)
                       segment-count 20]
                   {:bsdc-seg/N dimensions
                    :bsdc-seg/segment-count segment-count
                    :bsdc-seg/segment-length
                    (/ dimensions segment-count)})))
    (let [address-count (long 1e4)
          word-length (:bsdc-seg/N hd/default-opts)
          address-density 0.005
          decoder-threshold 2
          state
          {:content-matrix (->content-matrix address-count
                                             word-length)
           :decoder (->k-fold-address-decoder
                     {:address-count address-count
                      :address-density address-density
                      :k-delays 5
                      :word-length word-length})}
          ;; T (repeatedly 1e3 #(hd/->hv))
          char->t (into {}
                        (map (fn [g] [g (hd/->hv)]))
                        ;; a,b,c,...z
                        (map char (range 97 123)))
          history (atom [])]
      (doseq [data (take 5 (map val char->t))]
        (swap! history conj
               (-> state :decoder get-state))
        (write! (:content-matrix state)
                (decode-and-step! (:decoder state) data decoder-threshold)
                data))
      (clear-activations (:decoder state))
      [char->t
       (read-sequence! state
                       (get char->t \a)
                       decoder-threshold)]))

  (def out *1)


  (for [query-v (second out)]
    (let [m (into [] (map val (first out)))
          query-v (if-not (dtt/tensor? query-v)
                    (torch->jvm query-v)
                    query-v)
          threshold 0]
      (let [similarities
            (into [] (pmap #(hd/similarity % query-v) m))]
        (when (seq similarities)
          (let [argmax (dtype-argops/argmax similarities)]
            (when (<= threshold (similarities argmax))
              ((clojure.set/map-invert (first out))
               (m argmax))))))))
  '(\a \a \a \c \c \d)







  (do
    (do (System/gc) (py.. torch/cuda empty_cache))
    (alter-var-root #'hd/default-opts
                    (constantly
                     (let [dimensions (long 1e4)
                           segment-count 20]
                       {:bsdc-seg/N dimensions
                        :bsdc-seg/segment-count segment-count
                        :bsdc-seg/segment-length
                        (/ dimensions segment-count)})))
    (let [address-count (long 1e3)
          word-length (:bsdc-seg/N hd/default-opts)
          address-density 0.03
          decoder-threshold 2
          state {:content-matrix (->content-matrix
                                  address-count
                                  word-length)
                 :decoder (->k-fold-address-decoder
                           {:address-count address-count
                            :address-density address-density
                            :k-delays 5
                            :word-length word-length})}
          ]

      (def history (atom []))
      (doseq [[_ data] (take 5 T)]
        (swap! history conj (-> state :decoder get-state))
        (auto-associate! (:content-matrix state) data (:decoder state) decoder-threshold))

      ;; @history clear! In physiology, one might do this
      ;; by querying with 'nothing' for a few times
      (clear-activations (:decoder state))
      (def thestate state)

      (let [c (rand-nth (into [] (map char) (range 97 123)))
            c-t (get T c)
            read-and-step!
            (fn [address]
              ;; (def address address)
              ;; (def state state)
              ;; (def decoder-threshold decoder-threshold)
              (let
                  [addresses
                   (decode (:decoder state) address decoder-threshold)
                   out1 (sdm-read (:content-matrix state) addresses 1)]
                ;; (torch->jvm (:result out1))
                  out1))


            ;; outcomes
            ;; query 3 times
            ;; (reductions (fn [address _] (read-and-step! address)) c-t (range 2))
            ]
        ;; (count outcomes)
        ;; (map #(cleanup T %)  outcomes)

        ;; (let [out1 (read-and-step! (get T \a))
        ;;       out2 (read-and-step! (torch->jvm (:result out1)))
        ;;       out3 (read-and-step! (torch->jvm (:result out2)))]
        ;;   (map #(cleanup T %) [out1 out2 out3]))

        (def T T)
        (let [out1 (read-and-step! (get T \a))
              out2 (read-and-step! (torch->jvm (:result out1)))]
          [(cleanup T (torch->jvm (:result out2)))
           out2]))))




  @history

  ;; should be 'b'
  (cleanup T
           (torch->jvm
            (:result (sdm-read (:content-matrix thestate)
                               (indices->address-locations
                                (long 1e3)
                                #{130 468 676 723 890 402
                                  954 515 419 944 498 528
                                  303 522 456 411 201 489 47
                                  533 16 288 73 633 744})
                               1))))














  (clear-activations (:decoder state))

  (let [read-and-step!
        (fn [address]
          ;; (def address address)
          ;; (def state state)
          ;; (def decoder-threshold decoder-threshold)
          (let
              [addresses
               (decode (:decoder state) address decoder-threshold)
               out1
               (sdm-read (:content-matrix state) addresses 1)]
              (torch->jvm (:result out1))))]
    (let [out1 (read-and-step! (get T \a))
          out2 (read-and-step! out1)
          out3 (read-and-step! out2)]
      (map #(cleanup T %) [out1 out2 out3])))


  (cleanup T (first (rand-nth (into [] T))))


  (ffirst (sort-by second
                   (fn [a b]
                     (compare
                      (hd/similarity b (get T \e))
                      (hd/similarity a (get T \e))))
                   (into [] T))))








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

;; 5
;; https://nextjournal.com/cdeln/reference-counting-in-clojure
;;
