
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
