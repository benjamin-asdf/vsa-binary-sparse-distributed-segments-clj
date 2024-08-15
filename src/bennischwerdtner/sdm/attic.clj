
(comment
  (def a (hd/->seed))
  (def a-t (to-torch a))
  (def b (hd/->seed))
  (def b-t (to-torch b))
  (do (auto-associate! content-matrix address-matrix b-t)
      nil)
  (def c (hd/thin (hd/bundle a b)))
  (def c-t (to-torch c))
  (do (auto-associate! content-matrix address-matrix a-t)
      (auto-associate! content-matrix address-matrix b-t)
      (auto-associate! content-matrix address-matrix c-t)
      nil)
  (hd/maximally-sparse?
   (dtt/ensure-tensor (py.. (sdm-read content-matrix
                                      (decode-addresses
                                       address-matrix
                                       a
                                       decoder-threshold)
                                      1)
                            (to "cpu")
                            (numpy))))
  (def outc
    (dtt/ensure-tensor (py.. (sdm-read content-matrix
                                       (decode-addresses
                                        address-matrix
                                        c-t
                                        decoder-threshold)
                                       2)
                             (to "cpu")
                             (numpy))))
  ;; outc now has both a and b inside it
  (hd/similarity (hd/thin outc) c)
  (hd/similarity (hd/thin outc) a)
  (hd/similarity (hd/thin outc) b)
  ;; if you say top-k = 2, you get a and b _completely_
  ;; out of this why does it work like that? Because
  ;; there is enough overlap in the addresses of c with
  ;; both a and b
  (hd/similarity outc b)
  (hd/similarity outc a)
  (f/sum outc)
  ;; 200.0 sparsity x 2
  ;;
  ;;
  ;; This makes sparsity realy cool. Because we can
  ;; either mix and and thin, sharpening to a point
  ;; between a and b, or mix and 'accumulate' density,
  ;; and then represent the 'actual sumset' of a and b
  ;;
  ;; So mixing comes in 2 flavors.
  ;;
  ;; ->. With an sdm at hand, if you want to say that 2
  ;; things are associated, then auto associate the mix
  ;; of a and b
  ;;
  (def outa
    (dtt/ensure-tensor (py.. (sdm-read content-matrix
                                       (decode-addresses
                                        address-matrix
                                        a-t
                                        decoder-threshold)
                                       2)
                             (to "cpu")
                             (numpy))))
  ;;
  ;; this is now more similar to c than to a
  ;; that is random from the thinning though
  ;;
  (hd/similarity (hd/thin outa) c)
  (hd/similarity (hd/thin outa) a)
  (hd/similarity (hd/thin outa) b)
  (hd/similarity outa a)
  (hd/similarity outa c)
  (hd/similarity outa outc)
  (def outa
    (dtt/ensure-tensor (py.. (sdm-read content-matrix
                                       (decode-addresses
                                        address-matrix
                                        a-t
                                        decoder-threshold)
                                       1)
                             (to "cpu")
                             (numpy))))
  ;;
  ;; ... if you want a 'sharp' answer, use
  ;; (read-k = 1)
  ;;
  [(hd/similarity (hd/thin outa) c)
   (hd/similarity (hd/thin outa) a)
   (hd/similarity (hd/thin outa) b) (hd/similarity outa a)
   (hd/similarity outa c) (hd/similarity outa outc)]
  [0.46 1.0
   ;; a and b are dissimilar
   0.0 1.0 0.46 1.0])


(comment
  (do
    (def word-length (:bsdc-seg/N hd/default-opts))
    (def address-length word-length)
    (def address-count (long 1e4))
    (def address-density-k 6)
    ;; G
    (def decoder-threshold 2)
    (def decoder
      (->address-decoder
       {:address-count address-count
        :address-density (/ 6 1e4)
        ;; (/ 12 1e4)
        :word-length word-length}))
    (def content-matrix
      (->content-matrix address-count word-length))
    (def T
      (into []
            (map (fn [_] (hd/->seed))
                 (range 1e3)
                 ;; (range 1e4)
                 )))
    ;;
    ;; 10k items is half a minute on my machine now
    ;; datatransfers make 5 seconds
    ;; decoder dominates the time
    ;;
    (let [tseq
          ;; (time (into []
          ;;             (map
          ;;             #(pyutils/ensure-torch
          ;;                    %
          ;;                    torch-device)
          ;;                  T
          ;;                  ;; (take 100 T)
          ;;                  )))
          T]
      (time
       (doseq [t (take 100 tseq)]
         (let [t (pyutils/ensure-torch t torch-device)]
           ;; (decode decoder t decoder-threshold)
           (write! content-matrix
                   (decode decoder t decoder-threshold)
                   t))
         (py.. torch/cuda synchronize)))))

  (py.. torch/cuda empty_cache)
  (py/attr-type-map torch/cuda)

  (let [t (first T)]
    (hd/similarity
     t
     (:result
      (lookup-iteratively
       content-matrix
       (pyutils/ensure-torch (hd/weaken t 0.8) torch-device)
       (fn [address decoder-threshold]
         (decode decoder address decoder-threshold))
       (merge hd/default-opts
              {:decoder-threshold 2 :top-k 1})))))

  (def outcomes
    (time
     (doall
      (for [t T]
        (let [t (hd/weaken t 0.8)
              outcome (lookup-iteratively
                       content-matrix
                       t
                       (fn [address decoder-threshold]
                         (decode decoder
                                 address
                                 decoder-threshold))
                       (merge hd/default-opts
                              {:decoder-threshold 2
                               :top-k 1}))]
          (merge outcome
                 {:similarity (hd/similarity t
                                             (:last-outcome
                                              outcome))}
                 {:t t}))))))

  (f/descriptive-statistics (map :similarity outcomes))
  {:max 1.0
   :mean 0.929709
   :min 0.02
   :n-elems 10000
   :standard-deviation 0.15282055074849354}
  (count (filter #(< % 0.9) (map :similarity outcomes)))
  (count (filter #(< % 0.1) (map :similarity outcomes)))
  ;; those are the ones that trule fail,
  16
  ;; if similarity > 0.1 , reader would still recognize
  ;; their input as similar
  ;;
  (f/descriptive-statistics (map :step outcomes))
  {:max 5.0
   :mean 1.5129
   :min 1.0
   :n-elems 10000
   :standard-deviation 1.13717320857051}
  (count (filter #(< 1 %) (map :step outcomes)))
  2246
  (count (filter #(= 2 %) (map :step outcomes)))
  977
  (count (filter #(= 3 %) (map :step outcomes)))
  374
  (count (filter #(= 4 %) (map :step outcomes)))
  176
  (count (filter #(= 5 %) (map :step outcomes)))
  ;; fair to say that these did not converge?
  ;; everything above converged
  719)


(comment
  (defn ->numpy-indices
    [length indices]
    (let [t (numpy/zeros [length] numpy/int8)]
      (py/set-item!
       t
       (dtt/->tensor indices {:datatype :int32})
       1)
      t))

  (defn bool-tens->jvm-indices [tens]
    (tech.v3.datatype.unary-pred/bool-reader->indexes
     (dtt/ensure-tensor
      (py.. tens
        (to "cpu" :dtype torch/int8)
        (numpy)))))



  (do
    (def k-delays 6)
    (def word-length (:bsdc-seg/N hd/default-opts))
    (def address-length word-length)
    (def address-count (long 1e4))
    ;; higher density for k-fold memory
    (def address-density-k (* 6 k-delays))
    ;; G
    (def decoder-threshold 2)
    (def address-matrix
      (->address-matrix
       address-count
       address-length
       (/ address-density-k address-length)))
    (def content-matrix
      (->content-matrix address-count word-length))


    (defn to-torch [tens]
      (let [t-numpy (numpy/zeros [word-length] :dtype numpy/int8)]
        (dtt/tensor-copy! tens (dtt/ensure-tensor t-numpy))
        (torch/tensor t-numpy
                      :dtype torch/float16
                      :device torch-device)))



    (def delays (->address-delays address-count k-delays))
    (def decoder-state (atom (->address-state delays)))
    ;; write:
    (def T (into [] (map (fn [_] (hd/->seed)) (range 5)))))


  ;; t0

  (let
      [address-locations (decode-addresses address-matrix
                                           (to-torch (first
                                                      T))
                                           2)
       decoder-state (swap! decoder-state with-activations
                            (bool-tens->jvm-indices
                             address-locations))]
      (write! content-matrix
              (torch/tensor (dtt/->tensor
                             (k-fold-active-locations
                              decoder-state)))
              (to-torch (second T))))

  (hd/similarity
   (second T)
   (torch->jvm
    (sdm-read
     content-matrix
     (decode-addresses
      address-matrix
      (to-torch (first T))
      decoder-threshold)
     1)))

  (defn k-fold-write-1!
    [decoder-state address content]
    (let [address (to-torch address)
          content (to-torch content)
          address-locations (decode-addresses
                             address-matrix
                             address
                             decoder-threshold)
          decoder-state (with-activations
                          decoder-state
                          (bool-tens->jvm-indices
                           address-locations))]
      (write! content-matrix
              (torch/tensor (dtt/->tensor
                             (k-fold-active-locations
                              decoder-state)))
              content)
      decoder-state))

  (reset! decoder-state
          (k-fold-step
           (k-fold-write-1! @decoder-state (first T) (second T))))

  (reset!
   decoder-state
   (k-fold-step (k-fold-write-1!
                 @decoder-state
                 (nth T 1)
                 (nth T 2))))



  (reset!
   decoder-state
   (reduce
    (fn [decoder-state [a b]]
      (->
       (k-fold-write-1! decoder-state a b)
       k-fold-step))
    @decoder-state
    (map vector T
         (drop 1 T))))

  (defn k-fold-read
    [decoder-state address content-matrix top-k opts]
    (let [address (to-torch address)
          address-locations (decode-addresses
                             address-matrix
                             address
                             decoder-threshold)
          decoder-state (with-activations
                          decoder-state
                          (bool->jvm-indices
                           address-locations))
          active-locations (k-fold-active-locations
                            decoder-state)
          read-result (sdm-read
                       content-matrix
                       (->numpy-indices address-count active-locations)
                       ;; (torch/tensor
                       ;;  (->numpy-indices
                       ;;   address-count
                       ;;   active-locations)
                       ;;  :dtype torch/int8
                       ;;  :device
                       ;;  torch-device)
                       top-k)]
      {:decoder-state decoder-state
       :read-result read-result}))

  (def outcome
    (k-fold-read
     (clear-activations @decoder-state)
     (first T)
     content-matrix
     1
     {}))

  (hd/similarity (second T) (torch->jvm (:read-result outcome)))
  (k-fold-active-locations (:decoder-state outcome)))


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

(defn mem-sdm
  [{:keys [address-count word-length address-density]}]
  (let [addresses (atom {})
        content-matrix (->content-matrix address-count word-length)

        ]
    (reify
      SDM
        (decode-address [this address decoder-threshold]

          (torch/tensor
           (into
            []
            (or (get @addresses address)
                (let [addrs (into #{}
                                  (repeatedly
                                   10
                                   #(fm.rand/irand
                                     0
                                     address-count)))]
                  (swap! addresses assoc address addrs)
                  addrs)))
           :dtype
           torch/long))
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



(comment
  (binding [torch-device :cpu]
    (do (alter-var-root
          #'hd/default-opts
          (constantly (let [dimensions 25]
                        {:bsdc-seg/N dimensions
                         :bsdc-seg/segment-count
                           segment-count
                         :bsdc-seg/segment-length
                           (/ dimensions segment-count)})))
        (let [m (mem-sdm {:address-count 100
                          :address-density 0.2
                          :word-length 25})
              d (hd/->hv)]
          (decode-address m (hd/->hv) 2)
          (write m d d 2)
          (lookup m d 1 2)))))

(comment
  (binding [torch-device :cpu]
    (let [addr (torch/tensor [0 1 1 0 1] :dtype torch/float16)
          word (torch/tensor [1 1 0 0 1] :dtype torch/float16)
          C (torch/sparse_coo_tensor :size [5 5]
                                     :device torch-device
                                     :dtype torch/float32)
          counter-max 3]
      (let [activated-locations (py.. (torch/nonzero addr)
                                      (view -1))
            word-nonzero (py.. (torch/nonzero word) (view -1))
            indices
            (py..
             (torch/cartesian_prod activated-locations
                                   word-nonzero)
             (t))
            values (torch/ones (py.. indices (size 1)))
            update (torch/sparse_coo_tensor indices values (py.. C size))
            C
            ;; (to_dense)

            (py.. C (add_ update) (coalesce))

            ]
        ;; (py.. C (add_ update))
        ;; (py.. update (to_dense))
        ;; (py.. C
        ;;   (add_ update)
        ;;   (coalesce)
        ;;   (to_dense))
        (py.. C values (clamp_ 0 counter-max))

        (py.. C (to_dense)))))


  )


(comment


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
        (update :t inc))))



(comment

  (require-python '[builtins])
  (def t (torch/randn [5 3]))
  [t (py/get-item
      t
      [[0 1]
       (builtins/slice nil)])]

  [t
   (py/get-item t
                [(torch/tensor [true false true false false]
                               :dtype
                               torch/bool)
                 (builtins/slice nil)])]

  )

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


(defn ->delayed-address-decoder-1
  [{:keys [address-count word-length address-density
           k-delays]}]
  (let [address-matrix (->address-matrix-coo
                        address-count
                        word-length
                        address-density)
        address-delays (->address-delays
                        {:address-count address-count
                         :k-delays k-delays})
        make-state (fn []
                     {:t 0
                      ;; the future addresses
                      :delay-table (delay-activation-table
                                    {:address-count
                                     address-count
                                     :k-delays k-delays})})
        state (atom (make-state))]
    (reify
      clojure.lang.IDeref
      (deref [this] @state)
      Resets
      (reset [this]
        ;; t -> 0
        ;; no delays remembered anymore
        (reset! state (make-state)))
      AddressDecoder
      (decode-address [this address decoder-threshold]
        (let [address-locations (decode-addresses-coo
                                 address-matrix
                                 address
                                 decoder-threshold)
              {:keys [delay-table t]} @state
              delay-table (write-delay-table!
                           delay-table
                           address-locations
                           address-delays)
              now-location-set (read-delay-table
                                delay-table)
              next-table (update-delay-table delay-table)
              _ (reset! state {:delay-table next-table
                               :t (inc t)})]
          now-location-set)))))

(comment
  (def decoder
    (->delayed-address-decoder
     {:address-count 5
      :word-length 25
      :address-density 0.5
      :k-delays 3}))


  (def a (hd/->hv))

  (decode-address decoder a 2)

  ;; [[True False False]
  ;;  [False False False]
  ;;  [False True False]
  ;;  [False True False]
  ;;  [False False False]]

  (-> @decoder :delay-table (read-delay-table))
  ;; tensor([ True, False, False, False, False])
  )


(comment

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

  1.0


  ;; M = 1e6, N = 1e4, addr-density = 0.0005
  ;; Kinda cool that I can do it with 1 million locations
  ;; in some ways mirroring the efficiency of the brain sparse endoding
  ;; (only that here its storage and in brain its energy)



  ;; The memory requirements for the coo matrix is roughly

  (defn c-requirement
    [address-locations-per-input T bit-per-input]
    (let [nse (* address-locations-per-input T bit-per-input)
          ndim 2
          itemsize (py.. torch/uint8 -itemsize)]
      (* nse (+ (* ndim 8) itemsize))))

  (defn to-mib [bytes]
    (/ bytes (Math/pow 2 20)))
  (to-mib (c-requirement 40 (long 1e3) 20))

  ;; M = 1e6

  (binding [*torch-device* :cuda]
    (do (alter-var-root
         #'hd/default-opts
         (constantly (let [dimensions (long 1e4)
                           segment-count 20]
                       {:bsdc-seg/N dimensions
                        :bsdc-seg/segment-count segment-count
                        :bsdc-seg/segment-length
                        (/ dimensions segment-count)})))
        #_(def T
            (into []
                  (map #(pyutils/ensure-torch % :cuda)
                       (repeatedly 1e3 #(hd/->hv)))))
        (def T
          (into []
                (map pyutils/ensure-torch
                     (repeatedly 2 #(hd/->hv)))))
        (time (let [m (sparse-sdm {:address-count (long 1e6)
                                   :address-density 0.00095
                                   :word-length (long 1e4)})]
                ;; (doseq [[idx t] (map-indexed vector
                ;;                              (take
                ;;                              1000 T))]
                ;;   (write m t t 2))
                ;; (let [d (first T)]
                ;;   (hd/similarity
                ;;     (torch->jvm d)
                ;;     (torch->jvm (:result (lookup m d 1
                ;;     2)))))
                (torch/sum (decode-address m (first T) 2))))))


  (binding [*torch-device* :cuda]
    (do (alter-var-root
         #'hd/default-opts
         (constantly (let [dimensions (long 1e4)
                           segment-count 20]
                       {:bsdc-seg/N dimensions
                        :bsdc-seg/segment-count segment-count
                        :bsdc-seg/segment-length
                        (/ dimensions segment-count)})))
        #_(def T
            (into []
                  (map #(pyutils/ensure-torch % :cuda)
                       (repeatedly 1e3 #(hd/->hv)))))
        (def T
          (into []
                (map pyutils/ensure-torch
                     (repeatedly 1e3 #(hd/->hv)))))
        (time (let [m (sparse-sdm {:address-count (long 1e5)
                                   :address-density 0.0014
                                   :word-length (long 1e4)})]
                (doseq [t T]
                  (write m t t 2))
                (let [d (rand-nth T)]
                  (hd/similarity (torch->jvm d)
                                 (torch->jvm
                                  (:result
                                   (lookup m d 1 2)))))))))
  1.0

  ;; this config is fast I guess, 5.5s for 1000 writes
  ;; but is has very low address count for random addr., so presumably not much capacity

  (torch/sum
   (decode-address (->decoder-coo
                    {:address-count (long 1e5)
                     :address-density 0.0014
                     :word-length (long 1e4)})
                   (hd/->hv)
                   2))

  (time
   (torch/sum
    (decode-address (->decoder-coo
                     {:address-count (long 1e5)
                      :address-density 0.00003
                      :word-length (long 1e4)})
                    (hd/->hv)
                    1)))


  ;; depends heavily on :address-density,
  ;; I don't have enough gpu memory to go to a good density, which would be given by [[ideal-p]]
  ;; then the count of address would be  ~ 360, for T = 1e4, M = 1e6
  ;;


  ;; "Elapsed time: 29745.362644 msecs"


  (float (/ (* (/ 37 1000) 10000) 60))
  6.1666665
  ;; if you want to deal with 10k hypervectors, you you would need to bring 5-10min with this implementation
  ;;
  ;; if you want to deal with 1k hypervectors, that's too long for enjoying the repl
  ;;
  ;; if you want to deal with 100 hypervectors, it's fast.
  ;;


  ;; M = 1e5 works on cpu
  ;; can't really recommend it.
  ;; interesting experience for me to feel the power of parallel processing.
  ;;

  (binding [*torch-device* :cpu]
    (do (alter-var-root
         #'hd/default-opts
         (constantly (let [dimensions (long 1e4)
                           segment-count 20]
                       {:bsdc-seg/N dimensions
                        :bsdc-seg/segment-count
                        segment-count
                        :bsdc-seg/segment-length
                        (/ dimensions segment-count)})))
        (def T (into [] (map pyutils/ensure-torch (repeatedly 1e3 #(hd/->hv)))))
        (time
         (let [m (sparse-sdm {:address-count (long 1e5)
                              :address-density 0.002
                              :word-length (long 1e4)})]
           (doseq [t (take 100 T)]
             ;; (decode-address m t 2)
             (write m t t 2))
           (let [d (first T)]
             (write m d d 2)
             (lookup m d 1 2)
             (hd/similarity
              (torch->jvm d)
              (torch->jvm (:result (lookup m d 1 2)))))))))
  ;; 4s
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

(comment
  (py/set-item! (torch/zeros [5 5]) [0] 1)
  ;; >
  (py/set-item! (torch/zeros [5 5]) [0 1] 1)
  ;; >
  ;;  v
  (py/set-item! (torch/zeros [5 5]) [0 1 2] 1)
  ;; error dim >
  ;;  v > !
  (py/set-item! (torch/zeros [5 5]) [[0 1]] 1)
  ;; > >
  (py/set-item! (torch/zeros [5 5]) [[0 1] [0 1]] 1)
  ;; >
  ;;  v
  ;; >
  ;;  v
  ;; goal:
  ;; ----
  ;; >
  ;;  v ( one of randint 6 )
  ;; ... 1e6
  ;;
  (let [t (torch/zeros [3 5])]
    (let [idxs [(torch/arange (py.. t (size 0)))
                (torch/randint :low 0
                               :high (py.. t (size 1))
                               :size [(py.. t (size 0))])]]
      [idxs (py/set-item! t idxs 1)]))

  )

(comment
  (alter-var-root
   #'hd/default-opts
   #(merge %
           (let [dimensions 25
                 segment-count 5]
             {:bsdc-seg/N dimensions
              :bsdc-seg/segment-count segment-count
              :bsdc-seg/segment-length
              (/ dimensions segment-count)})))




  (alter-var-root
   #'hd/default-opts
   #(merge %
           (let [dimensions (long 1e4)
                 segment-count 20]
             {:bsdc-seg/N dimensions
              :bsdc-seg/segment-count segment-count
              :bsdc-seg/segment-length
              (/ dimensions segment-count)})))




  (alter-var-root
   #'hd/default-opts
   #(merge %
           (let [dimensions (long 1e4)
                 segment-count 100]
             {:bsdc-seg/N dimensions
              :bsdc-seg/segment-count segment-count
              :bsdc-seg/segment-length
              (/ dimensions segment-count)})))


  hd/default-opts
  {:bsdc-seg/N 10000
   :bsdc-seg/segment-count 20
   :bsdc-seg/segment-length 500
   :tensor-opts {:container-type :native-heap}})

(comment


  (torch/sum
   (decode-address (->decoder-coo
                    {:address-count (long 1e5)
                     :address-density 0.002
                     :word-length (long 1e4)})
                   (hd/->hv)
                   2))

  ;; (/ 180 6)

  (torch/sum
   (decode-address (->decoder-coo
                    {:address-count (long 1e6)
                     :address-density 0.00001
                     :word-length (long 1e4)})
                   (hd/->hv)
                   1))


  (torch/sum
   (decode-address (->decoder-coo
                    {:address-count (long 1e6)
                     :address-density 0.00001
                     :word-length (long 1e4)})
                   (hd/->hv)
                   1))


  )




























;; ----------------------------------------------------
;; codebook decoding

(def a (hd/->seed))

(let
    [vocabulary-matrix
     (py.. (torch/ge (torch/tensor (/ 20 (long 1e4))
                                   :device
                                   *torch-device*)
                     (torch/randn [10 (long 1e4)]
                                  :device
                                  *torch-device*))
       (to :dtype torch/float16))]
    (torch/argmax (torch/mv vocabulary-matrix
                            (py.. (pyutils/ensure-torch a)
                              (to :dtype
                                  torch/float16)))))


(torch/argmax (torch/randn [4 4]) :dim 1)

(torch/argmax (torch/randn [4]))

(time (dotimes [n 10000]
        (hd/->seed)))
;; "Elapsed time: 211.294884 msecs"

(defn preallocated-vocabulary
  [n]
  (dtt/->tensor (repeatedly n hd/->seed) :datatype :int8))

(:shape (dtt/tensor->dimensions (preallocated-vocabulary 100)))

(time (let [alphabet-jvm (preallocated-vocabulary 1000)
            codebook-matrix (py.. (pyutils/ensure-torch
                                    alphabet-jvm)
                                  (to :dtype torch/float16))
            a (rand-nth vocab-jvm)]
        (torch/select
          codebook-matrix
          0
          (torch/argmax
            (torch/mv vocabulary-matrix
                      (py.. (pyutils/ensure-torch a)
                        (to :dtype torch/float16)))))))

(let [alphabet-jvm (preallocated-vocabulary 1000)
      codebook-matrix (py.. (pyutils/ensure-torch
                              alphabet-jvm)
                            (to :dtype torch/float16))]
  (def alphabet-jvm alphabet-jvm))




;; ----------------------------------
;;
;; Resonator network







































;; --------------------------------------------
;; Neuronal Ensemble Area
;;

;; 1. Random directed graph, log distributed (Buzsáki)
;;
;; 2. Learning rule: Intrinsic excitability (Yuste)
;;
;; 3. Inhibition model?
;; - top-k per segment (modeling sheet inhibition)
;; - long range inhibition? model recruitment of basket cells?
;;








(comment
  (decode-addresses (->address-matrix 3 3 0)
                    (torch/tensor [0 1 1]
                                  :dtype torch/float16
                                  :device *torch-device*)
                    1)
  (decode-addresses-coo (->address-matrix-coo 3 3 0)
                        (torch/tensor [0 1 1]
                                      :dtype torch/float32
                                      :device
                                      *torch-device*)
                        1)
  (decode-addresses-coo (->address-matrix-coo 3 3 0.5)
                        (torch/tensor [0 1 1]
                                      :dtype torch/float
                                      :device
                                      *torch-device*)
                        1)
  (py.. (torch/tensor [0 1 1]
                      :dtype torch/float
                      :device *torch-device*)
        -dtype)
  (decode-addresses-coo
   (->address-matrix-coo 3 3 0.5)
   (torch/tensor
    [[0 1 1]
     [0 1 1]]
    :dtype torch/float
    :device *torch-device*)
   1)
  (torch/bmm
   (let
       [w (torch/tensor
           [[0 0 1]
            [0 1 1]]
           :dtype torch/float)]
       (torch/))
   ;; with batch dimension?
   (torch/tensor [[0 1 1]
                  [0 1 1]]
                 :dtype torch/float
                 :device *torch-device*))




  (comment
    (py.. (->content-matrix-coo 10 10) (to_dense))
    (require-python '[updateC :as pcoo])
    (alter-var-root #'pyutils/*torch-device*
                    (constantly :cpu))
    (pcoo/update_matrix_C
     (->content-matrix-coo 3 3)
     (torch/tensor [[true true false] [true true false]]
                   :dtype torch/bool
                   :device *torch-device*)
     (torch/tensor [[1 1 1] [1 0 1]] :device *torch-device*))
    (let [activated-locations
          (torch/nonzero (torch/tensor [true true false]))
          word-nonzero (py.. (torch/nonzero (torch/tensor
                                             [0 1 0]))
                             (view -1))]
      (torch/cartesian_prod activated-locations word-nonzero))
    (torch/cartesian_prod
     (torch/nonzero (torch/tensor [true true false]))
     ;; tensor([[0], [1]])
     (torch/nonzero (torch/tensor [0 1 0])))
    (py.. (->content-matrix-coo 3 3) indices)
    (torch/mul (torch/tensor [true true false])
               (torch/tensor [0 1 0]))
    (torch/mul (torch/tensor [true true false])
               (torch/tensor [0 1 0]))
    (py.. (torch/sparse_coo_tensor
           (py.. (torch/tensor [[0 0] [0 1]]) (t))
           (torch/ones [2])
           [3 3])
          (to_dense))
    ;; tensor([[1., 1., 0.],
    ;;         [0., 0., 0.],
    ;;         [0., 0., 0.]])
    (let [y (torch/tensor [[true true false]
                           [false false true]])
          w (torch/tensor [[0 1 0] [1 1 1]])]
      (torch/mv w y))
    (let [w (->content-matrix-coo 3 3)
          w (write-coo!
             w
             (torch/tensor [true true false]
                           :dtype torch/bool
                           :device *torch-device*)
             (torch/tensor [1 1 1] :device *torch-device*))
          w (write-coo!
             w
             (torch/tensor [true true false]
                           :dtype torch/bool
                           :device *torch-device*)
             (torch/tensor [1 0 1] :device *torch-device*))]
      (py.. w (to_dense)))
    ;; tensor( [[2, 1, 2],
    ;;          [2, 1, 2],
    ;;          [0, 0, 0]], device='cuda:0',
    ;;          dtype=torch.uint8)
    (let [w (->content-matrix-coo 3 3)
          w (write-coo-batch!
             w
             (torch/tensor [[true true false]
                            [true true false]]
                           :dtype torch/bool
                           :device *torch-device*)
             (torch/tensor [[1 1 1] [1 0 1]]
                           :device
                           *torch-device*))]
      (py.. w (to_dense)))))
