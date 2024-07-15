(ns bennischwerdtner.sdm.sdm
  (:require
   [clojure.set :as set]
   [bennischwerdtner.pyutils :as pyutils]
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
;;                                             |         v     v    v        where S >= read-threshold
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
;;
;; Address decoder
;; ----------------
;;
;; - M hard address locations of N (address-word-lenght) width
;; - address locations are sparse 0s and 1s with address-density << 1
;; - Each address location models one granule cell, granule cells outnumber mossy fibers (their inputs) 200 to 1
;;   (https://en.wikipedia.org/wiki/Cerebellar_granule_cell#:~:text=Granule%20cells%20receive%20all%20of,a%20much%20more%20expansive%20way.)
;;
;;
;; Since addreses are not dense, this is a 'intermediate design' (Jaeckel, L.A. 1989b)
;;
;; In the address decoder step, we can massively use parallelism.
;;
;;
;; Storage / Content Matrix `C`
;; -----------------
;;
;; - M x U counters
;; - here, U == N, allowing for auto association. I.e. address-word == input-word.
;; - Different from Kanerva 1993, where each location is in range {-15...15},
;;   here range is {0..`counter-max`}, counter-max = 100 (?)
;;
;; Read (y):
;; - sum up counters of the y activated address content locations of C (-> S)
;; - Remove sums below read-threshold, reducing noise, trading signal strenght. In other words, this increases
;;   the count and strength of address locations for a stored word `a` to be retrieved.
;; - In Kanerva 1993: take the sign of the sums
;; - Here, take top-k (`read-k`) non zero bits for each word segment
;; - output vector `z` has read-k * segment-count non zero bits
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
;; M - memory count
;; T - (data set) should be 1-5% of M
;; p - probability of activation, ideally p = 0.000368
;; This is important, number of hard locations activated for an input
;; The best p maximizes signal to noise, is approx. 2MT^-1/3
;;
;;

(comment
  ;; what is p?
  ;; depends on the M the hard locations count,
  ;; density of the address matrix
  ;; density of address words
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
   2000000)

  (* 100 2.07E-4)




  )


;; ----------------------
;; Concrete torch implementation
;; ----------------------

(def ^:dynamic torch-device :cpu)

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
  (require '[libpython-clj2.python.np-array])
  (alter-var-root
   #'torch-device
   (constantly
    (if (py.. torch/cuda (is_available)) :cuda :cpu))))

(def counter-max 15)

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
                                torch-device)
                    density)
    (to torch/float16)))

(defn ->content-matrix
  [address-count word-length]
  (torch/zeros [address-count word-length]
               :device
               torch-device))

(defn decode-addresses
  [address-matrix address decoder-threshold]
  ;; d
  (let [address (pyutils/ensure-torch address torch-device)
        inputs (torch/mv address-matrix address)
        activations (torch/ge inputs decoder-threshold)]
    ;; y
    activations))

(comment
  (decode-addresses
   (->address-matrix 3 3 0)
   (torch/tensor [0 1 1] :dtype torch/float16 :device torch-device)
   1)
  ;; y
  ;; tensor([False,  True,  True], device='cuda:0')
  )


(defn write!
  [content-matrix address-locations input-word]
  (let [input-word (pyutils/ensure-torch input-word
                                         torch-device)]
    (py/set-item! content-matrix
                  address-locations
                  (py.. (py/get-item content-matrix
                                     address-locations)
                        (add_ input-word)
                        (clamp_ :min 0 :max counter-max)))))

(comment
  (write!
   (->content-matrix 3 3)
   ;; (torch/tensor [1 0 0] :dtype torch/long :device
   ;; torch-device)
   (torch/tensor [true false false] :dtype torch/bool)
   (torch/tensor [0 1 0]
                 :dtype torch/float16
                 :device torch-device))
  )

#_(defn sdm-read
  "



  `read-threshold`: Only take counter sums above read-threshold into account.
  Intuitively, imagine address `a` is stored and `b`, not stored, overlap in a single hard location from [[decode]].
  At low capacity, when most other bits are 0, querying with `b` results in `a`. Even though the address overlap was 1.

  Choosing the read-threshold should depend on `p`, the probability of activation for an address, and [[counter-max]].

  For p ~= 0.003872,
  address-count = 10.000,
  address-density = 0.0009,
  decoder-threshold = 2,
  counter-max = 15

  read-threshold = 20 gave good results.

  "
  ([content-matrix address-locations top-k read-threshold]
   (sdm-read content-matrix
             address-locations
             top-k
             read-threshold
             hd/default-opts))
  ([content-matrix address-locations top-k read-threshold
    {:bsdc-seg/keys [segment-count segment-length N]}]
   (py/with-gil
     (let [s (torch/sum (py/get-item content-matrix
                                     address-locations)
                        :dim
                        0)
           s (torch/where (torch/ge s read-threshold)
                          s
                          (torch/zeros_like s))
           summed (torch/reshape s
                                 [segment-count
                                  segment-length])
           ;; non-zero-mask (torch/any (torch/ne summed
           ;; 0) 1)
           indices (-> summed
                       (torch/topk (min top-k
                                        segment-length))
                       ;; (py.. -indices)
                   )
           result (torch/scatter (torch/zeros
                                   [segment-count
                                    segment-length]
                                   :dtype torch/uint8
                                   :device torch-device)
                                 1
                                 (py.. indices -indices)
                                 1)]
       (torch/reshape result [N])
       ;; (-> (torch/where non-zero-mask
       ;;                  result
       ;;                  (torch/zeros_like result))
       ;;     (torch/reshape [N]))
       ))))


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
   (py/with-gil
     (let [s (torch/sum (py/get-item content-matrix
                                     address-locations)
                        :dim
                        0)
           topk-result (-> s
                           (torch/reshape [segment-count
                                           segment-length])
                           (torch/topk
                             (min top-k segment-length)))
           result (torch/scatter (torch/zeros
                                   [segment-count
                                    segment-length]
                                   :dtype torch/uint8
                                   :device torch-device)
                                 1
                                 (py.. topk-result -indices)
                                 1)]
       ;; {:confidence
       ;;  (py.. (torch/sum address-locations) item)
       ;;  }
       {:confidence (torch/div
                      (torch/sum (py.. topk-result -values))
                      ;; also divide by top-k?
                      ;;
                      ;; scaling this with the
                      ;; address-locations count is
                      ;; probably taste. It turns out
                      ;; to say high confidence, if the
                      ;; count of address is low, when
                      ;; the address is a 'weak' (sub
                      ;; sparsity) hypervector. The
                      ;; caller would have to take this
                      ;; into account themselves.
                      ;;
                      (* segment-count
                         (py.. (torch/sum address-locations)
                               item)))
        :result (torch/reshape result [N])}))))

(comment

  (binding [torch-device :cpu]
    (let [{:keys [confidence result]}
          (sdm-read
           ;; content matrix
           (torch/tensor [[0 0 1 0]
                          [0 0 1 0]])
           ;; activations (y)
           (torch/tensor [true true])
           1
           {:bsdc-seg/N 4
            :bsdc-seg/segment-count 2
            :bsdc-seg/segment-length 2})]
      (and (torch/equal (torch/tensor 0.5) confidence)
           (torch/equal (torch/tensor [1 0 1 0]) result))
      ;; (torch/sum confidence)
      ))


  (binding
      [torch-device :cpu]
      (let [N (* 2 (inc (rand-int 5)))
            segment-count 2
            segment-length (int (/ N segment-count))
            {:keys [confidence result]}
            (sdm-read
             ;; content matrix
             (py.. (torch/ge (torch/rand [5 N]) (rand 0.5))
               (to :dtype torch/float16))
             ;; activations (y)
             (torch/ge (torch/randn [5]) 0.5)
             1
             {:bsdc-seg/N N
              :bsdc-seg/segment-count segment-count
              :bsdc-seg/segment-length segment-length})]
        ;; (and (torch/equal (torch/tensor 1) confidence)
        ;;      (torch/equal (torch/tensor [1 0 1 0])
        ;;      result))
        ;; (torch/sum confidence)
        [confidence result]))




















  (sdm-read
   (write!
    (->content-matrix 5 10)
    (torch/tensor
     [true true false false true]
     :dtype torch/bool
     :device torch-device)
    (torch/tensor
     [0 0 1 0 0
      1 0 0 1 0]
     :device torch-device))
   (torch/tensor
    [true true false false true]
    :dtype torch/bool
    :device torch-device)
   1
   {:bsdc-seg/N 10
    :bsdc-seg/segment-count 2
    :bsdc-seg/segment-length 5})



  (torch/topk
   (torch/tensor [[0 0 0] [0 0 0]])
   1)


  (py.. result size)
  ;; torch.Size([2, 5])
  (py.. non-zero-mask size)

  (torch/where non-zero-mask
               result
               (torch/zeros_like result))

  (torch/where
   (torch/tensor [true false])
   (torch/tensor [[2 4]
                  [3 4]])
   (torch/zeros [2]))


  (sdm-read
   (write!
    (->content-matrix 5 10)
    (torch/tensor
     [true true false false true]
     :dtype torch/bool :device torch-device)
    (torch/tensor
     [0 0 1 0 0
      1 0 0 1 0]
     :device torch-device))
   (torch/tensor
    [true true false false true]
    :dtype torch/bool
    :device torch-device)
   1
   0
   {:bsdc-seg/N 10
    :bsdc-seg/segment-count 2
    :bsdc-seg/segment-length 5}))

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


;; not sure yet
(defprotocol SDM
  (known?
    [this address]
    [this address decoder-threshold])
  (lookup
    [this address top-k]
    [this address top-k decoder-threshold])
  (converged-lookup
    [this address top-k]
    [this address top-k decoder-threshold])
  (write
    [this address content]
    [this address content decoder-threshold]))




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
  (dtt/clone (dtt/compute-tensor [address-count]
                                 (fn [_]
                                   (fm.rand/irand k-delays))
                                 :int8)))

;; this whole address bookeeping should not be so many since addresses are so sparse...
;; We can do this on jvm and upgrade when needed

(defn ->address-state
  [delay-index]
  {:delay-index delay-index
   :t 0
   ;; keeping track of the future
   :t->activations {}})

(defn with-activations
  [{:as state :keys [t delay-index t->activations]}
   activated-locations]
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

(defn k-fold-active-locations
  [{:keys [t t->activations]}]
  (t->activations t))

(defn k-fold-step
  [{:as state :keys [t]}]
  (-> state
      (update :t->activations dissoc t)
      (update :t inc)))

(defn clear-activations [state]
  (assoc state :t 0 :t->activations {}))


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
  (let [address (pyutils/ensure-torch address torch-device)]
    (write! content-matrix
            (decode decoder address decoder-threshold)
            address)))

(defn lookup-iteratively
  [content-matrix address decoder
   {:as opts
    :keys [decoder-threshold top-k read-threshold]}]
  ;; lookup iteratively, if you are within critical
  ;; distance this will converge to the output word
  ;; within a few steps
  ;; Kanerva 1988 [2]
  ;;
  (py/with-gil
    (reduce (fn [{:keys [last-outcome address]} step]
              (let [address (pyutils/ensure-torch
                             address
                             torch-device)
                    outcome (torch->jvm (sdm-read
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
            (range 6))))



(comment
  (do
    (System/gc)
    (py.. torch/cuda empty_cache))

  6327
  (py/get-item (py.. torch/cuda memory_stats) "active.all.current")
  4
  (py/get-item (py.. torch/cuda memory_stats) "active_bytes.all.current")
  (def t (torch/ones [1000] :device torch-device))
  (def t nil)
  6328)


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
    [(torch/sum addresses)
     (torch/sum
       (decode (:decoder state) t-prime decoder-threshold))
     (torch/sum
       (decode (:decoder state) tb decoder-threshold))]
    (auto-associate! (:content-matrix state)
                     t
                     (:decoder state)
                     decoder-threshold)
    (doseq
        [data T]
        (auto-associate! (:content-matrix state)
                         data
                         (:decoder state)
                         decoder-threshold))
    [(let [r (sdm-read (:content-matrix state) addresses 1)]
       [:sim-t-res
        (hd/similarity (torch->jvm (:result r)) t)
        :confidence (py.. (:confidence r) item)])
     ;; prime
     (let [r (sdm-read (:content-matrix state)
                       (decode (:decoder state)
                               t-prime
                               decoder-threshold)
                       1)]
       [:sim-t-prime-res
        (hd/similarity (torch->jvm (:result r)) t)
        :confidence (py.. (:confidence r) item)])
     (let [r (sdm-read (:content-matrix state)
                       (decode (:decoder state)
                               tb
                               decoder-threshold)
                       1)]
       [:sim-tb (hd/similarity (torch->jvm (:result r)) t)
        :confidence (py.. (:confidence r) item)])
     (let [r (sdm-read (:content-matrix state)
                       (decode (:decoder state)
                               tc
                               decoder-threshold)
                       1)]
       [:sim-tc (hd/similarity (torch->jvm (:result r)) t)
        :confidence (py.. (:confidence r) item)])]))


  [[:sim-t-res 1.0 :confidence 1.0]
   [:sim-t-prime-res 1.0 :confidence 1.0]
   [:sim-tb 1.0 :confidence 0.13636362552642822]
   [:sim-tc 0.0 :confidence 0.0]]

  [[:sim-t-res 1.0 :confidence 1.0]
   [:sim-t-prime-res 1.0 :confidence 1.0]
   ;; intermediate confidence when made from equal parts
   [:sim-tb 1.0 :confidence 0.3589743673801422]])



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
