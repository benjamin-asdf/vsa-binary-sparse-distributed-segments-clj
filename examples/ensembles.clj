(ns ensembles
  (:require
    [tech.v3.datatype.functional :as f]
    [tech.v3.datatype :as dtype]
    [tech.v3.tensor :as dtt]
    [tech.v3.datatype.bitmap :as bitmap]
    [fastmath.random :as fm.rand]
    [fastmath.core :as fm]
    [bennischwerdtner.sdm.sdm :as sdm]
    [bennischwerdtner.hd.binary-sparse-segmented :as hd]
    [bennischwerdtner.pyutils :as pyutils]
    [tech.v3.datatype.unary-pred :as unary-pred]
    [tech.v3.datatype.argops :as dtype-argops]
    [bennischwerdtner.hd.codebook-item-memory :as codebook]
    [bennischwerdtner.hd.ui.audio :as audio]
    [bennischwerdtner.hd.data :as hdd]
    [libpython-clj2.require :refer [require-python]]
    [libpython-clj2.python :refer [py. py..] :as py]))


;; ---------------------------------------
;; Re-cooking 'Assembly Calculus'
;;
;; paper: Assemblies of neurons learn to classify well-separated distributions
;; Max Dabagia, Christos H. Papadimitriou, Santosh S. Vempala
;;
;; https://arxiv.org/abs/2110.03171
;; ---------------------------------------
;;
;;
;; differences:
;; - I allow self-connection (https://en.wikipedia.org/wiki/Autapse)
;;

;; Notes:
;; - they refer to Buzsáki trajectories quite a bit,
;; - conceptually the analog in their model should be pre-allocated, robust 'scaffold' sequences
;;
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

(defn cap-k [inputs k]
  (py.. (torch/topk inputs k) -indices))

(defn update-activations
  [{:as state :keys [activations inputs N cap-k-k]}]
  (assoc state
         :last-activations activations
         :activations (py/set-item! (torch/zeros
                                     [N]
                                     :dtype torch/bool
                                     :device
                                     pyutils/*torch-device*)
                                    (cap-k inputs cap-k-k)
                                    1)))

(defn hebbian-plasticity
  [{:keys [last-activations activations weights
           hebbian-plasticity-beta]}]
  ;; for each j->i edge where j preceeds i firing,
  ;; increase
  ;;  * 1 + hebbian-plasticity-beta
  ;; ----------------------------------------------------
  ;; reference-implementation
  ;; (fn [{:keys [weights last-activations activations
  ;;              hebbian-plasticity-beta]}]
  ;;   (doseq [j (torch/nonzero last-activations)
  ;;           i (torch/nonzero activations)]
  ;;     (py/set-item!
  ;;      weights
  ;;      [j i]
  ;;      (* (py.. (py/get-item weights [j i]) item)
  ;;         (+ 1 hebbian-plasticity-beta))))
  ;;   weights)
  ;; ----------------------------------------------------
  (let [last-idx (torch/unsqueeze (torch/nonzero
                                    last-activations)
                                  1)
        idx (torch/nonzero activations)]
    (if (some (comp zero? #(py.. % nelement))
              [last-idx idx])
      weights
      (let [w (py.. weights clone)]
        (py/set-item! w
                      [last-idx idx]
                      (torch/multiply
                        (py/get-item w [last-idx idx])
                        (+ 1 hebbian-plasticity-beta)))
        w))))

(defn update-weights [state]
  (assoc state :weights (hebbian-plasticity state)))

(defn inputs [weights activations]
  (torch/sum
   (py/get-item
    weights
    ;;
    ;;
    ;; synapses j->i
    ;;    [[1.0000, 1.0000, 1.0000],   <-  j0
    ;;     [1.0500, 1.0000, 1.0000],       j1
    ;;     [1.0000, 1.0000, 1.0000]]       j2
    ;;
    ;;        i0       i1      i2
    ;;
    ;; inputs (activations):
    ;; [ false true true ]
    ;;
    ;;    [[1.0000, 1.0000, 1.0000],  _
    ;;     [1.0500, 1.0000, 1.0000],  <- on
    ;;     [1.0000, 1.0000, 1.0000]]  <- on
    ;;
    ;; inputs:
    ;;
    ;;  ->  2.05      2       2
    ;;
    ;;
    activations
    ;; sum the cols where the j is active rn
    ;; (j->i inputs)
    )
   0))


;; -------------------------------
;; topology
;;

(defn random-directed-graph
  [N density]
  (py.. (torch/le (torch/rand [N N]
                              :dtype torch/float
                              :device
                                pyutils/*torch-device*)
                  density)
    (to :dtype torch/float)))

(defn normalize-weights
  [weights]
  (torch/div weights (torch/sum weights 0 :keepdim true)))

;; log-normal distribution from
;; Buzsáki G, Mizuseki K. 2014
(defn random-directed-graph-log-normal [])


;; ----------------------------------------------------
;; book keeping
;;

(defn update-inputs
  [{:as state :keys [weights activations]}]
  (assoc state :inputs (inputs weights activations)))

(defn set-activations [state activations]
  (assoc state :activations activations))

(defn append-activations
  [state activations]
  (update state
          :activations
          (fn [current]
            (torch/bitwise_xor current activations))))

(defn ->neuronal-area
  [{:keys [N density hebbian-plasticity-beta cap-k-k]}]
  {:N N
   :activations (torch/zeros [N]
                             :dtype torch/bool
                             :device pyutils/*torch-device*)
   :cap-k-k cap-k-k
   :density density
   :hebbian-plasticity-beta hebbian-plasticity-beta
   :t 0
   :update-fns
   [(fn [state] (update state :t inc))
    ;; new inputs
    update-inputs
    ;; threshold
    update-activations
    ;; plasticity
    (fn [state]
      (assoc state :weights (hebbian-plasticity state)))
    ;; normalize every 15 times
    (fn [state]
      (if-not (zero? (mod (:t state) 15))
        state
        (update state :weights normalize-weights)))]
   :weights (random-directed-graph N density)})

(defn read-activations
  [state]
  (torch/squeeze (torch/nonzero (:activations state)) 1))

(defn update-area [{:as state :keys [update-fns]}]
  (reduce (fn [state op] (op state)) state update-fns))

(comment
  (let
      [a
       (->neuronal-area {:N (long 5)
                         :cap-k-k 2
                         :density 0.5
                         :hebbian-plasticity-beta 0.1})
       a (set-activations a (torch/le (torch/rand [5] :device pyutils/*torch-device*) 0.5))]

      [(update-inputs a)
       (update-activations (update-inputs a))
       (update-weights (update-activations (update-inputs a)))]))





(comment

  ;; Classifier
  ;; -------------------

  ;;
  ;; train an ensemble area:
  ;;

  (def input-classes
    (into {}
          (for [k [:a :b :c]]
            [k
             (torch/le (torch/rand [(long 1e3)]
                                   :device
                                   pyutils/*torch-device*)
                       ;; this is the projection
                       ;; probability, or related to it
                       ;;
                       0.05)])))

  (def neurons
    (atom (->neuronal-area {:N (long 1e3) :cap-k-k 100 :density 0.1 :hebbian-plasticity-beta 0.1})))

  (def input-classses->assemblies
    (into []
          (for [[k v] input-classes]
            (do (reset! neurons
                        (reduce (fn [neurons _]
                                  ;; repeated exposure will
                                  ;; form a stable cell
                                  ;; assembly
                                  (update-area
                                   (append-activations neurons v)))
                                (update-area (set-activations @neurons v))
                                (range 5)))
                {:assembly (:activations (update-area
                                          @neurons))
                 :class k
                 :sensor-data v}))))


  ;; now you can make an auto associative query
  ;;


  ;; query:
  ;;
  ;; ... but I can't query without modifying the net
  ;; (you probably want to make a mode that doesn't do plasticity)
  ;; (I *can* query and leave the net static because I made update immutable at the moment)
  ;; but the memory price is probably not toleratable. (1e3 neurons is tiny, but weight matrix get's large quick).
  ;; maybe with sparse tensors it would
  ;;


  ;;
  ;; book keeping, remember the assemblies
  ;;

  (def assembly-codebook
    (let [items (into [] input-classses->assemblies)
          assembly-book (py.. (torch/stack
                               (into []
                                     (map :assembly items)))
                          (to :dtype torch/float))]
      (fn lookup [q]
        (:class (nth items
                     (-> (torch/mv
                          assembly-book
                          (py.. q (to :dtype torch/float)))
                         (torch/argmax)
                         (py.. item)))))))


  ;; it learns to classify
  ;; -------------------------------------

  (for [drop-rate [0 0.25 0.5 0.75 1.0]
        runs (range 5)]
    (let [{:keys [class sensor-data assembly]}
          (rand-nth (into [] input-classses->assemblies))]
      ;; use a subset of sensor-data
      (let [mask (torch/ge (torch/rand [(long 1e3)]
                                       :device
                                       pyutils/*torch-device*)
                           drop-rate)
            sensor-data-prime (torch/bitwise_and mask
                                                 sensor-data)]
        (let [A
              ;; assembly
              (:activations (update-area
                             (set-activations
                              @neurons
                              ;; sensor-data
                              sensor-data-prime)))]
          ;; compare it to the well known assembly
          #_[class (torch/squeeze (torch/nonzero A) 1)
             (torch/squeeze (torch/nonzero assembly) 1)
             ;; ~ 0.8
             (torch/div
              (torch/sum (torch/bitwise_and A assembly))
              (py.. (torch/nonzero assembly) (nelement)))
             (assembly-codebook A)
             (= (assembly-codebook A) class)]
          {:class class
           :classified? (= (assembly-codebook A) class)
           :drop-rate drop-rate}))))

  '({:class :a :classified? true :drop-rate 0}
    {:class :b :classified? true :drop-rate 0}
    {:class :c :classified? true :drop-rate 0}
    {:class :b :classified? true :drop-rate 0}
    {:class :a :classified? true :drop-rate 0}
    {:class :b :classified? true :drop-rate 0.25}
    {:class :b :classified? true :drop-rate 0.25}
    {:class :a :classified? true :drop-rate 0.25}
    {:class :b :classified? true :drop-rate 0.25}
    {:class :c :classified? true :drop-rate 0.25}
    {:class :b :classified? true :drop-rate 0.5}
    {:class :c :classified? true :drop-rate 0.5}
    {:class :a :classified? true :drop-rate 0.5}
    {:class :a :classified? true :drop-rate 0.5}
    {:class :c :classified? true :drop-rate 0.5}
    {:class :c :classified? true :drop-rate 0.75}
    {:class :c :classified? true :drop-rate 0.75}
    {:class :b :classified? true :drop-rate 0.75}
    {:class :c :classified? true :drop-rate 0.75}
    {:class :c :classified? true :drop-rate 0.75}
    {:class :b :classified? true :drop-rate 1.0}
    {:class :b :classified? true :drop-rate 1.0}
    {:class :a :classified? false :drop-rate 1.0}
    {:class :b :classified? true :drop-rate 1.0}
    {:class :a :classified? false :drop-rate 1.0})



  ;; This time with mixing random noise
  ;; (do it 10 times and report mean success rate)

  (for [drop-rate [0 0.25 0.5 0.75 1.0]]
    (let [outcome
          (for [runs (range 10)]
            (let [{:keys [class sensor-data assembly]}
                  (rand-nth
                   (into [] input-classses->assemblies))]
              ;; use a subset of sensor-data
              (let [mask (torch/ge (torch/rand
                                    [(long 1e3)]
                                    :device
                                    pyutils/*torch-device*)
                                   drop-rate)
                    random-noise (torch/le
                                  (torch/rand
                                   [(long 1e3)]
                                   :device
                                   pyutils/*torch-device*)
                                  0.05)
                    sensor-data-prime
                    (torch/bitwise_and mask sensor-data)
                    sensor-data-prime (torch/bitwise_xor
                                       sensor-data-prime
                                       random-noise)]
                (let [A
                      ;; assembly
                      (:activations
                       (update-area
                        (set-activations
                         @neurons
                         sensor-data-prime)))]
                  {:class class
                   :classified? (= (assembly-codebook A)
                                   class)
                   :drop-rate drop-rate}))))]
      {:drop-rate drop-rate
       :success-rate (f/mean (map :classified? outcome))}))

  '({:drop-rate 0 :success-rate 1.0}
    {:drop-rate 0.25 :success-rate 1.0}
    {:drop-rate 0.5 :success-rate 0.8}
    {:drop-rate 0.75 :success-rate 0.6}
    {:drop-rate 1.0 :success-rate 0.3})


  ;; N = 1e3
  ;; beta = 0.1
  ;; k = 100
  ;; densitity = 0.1
  ;; projection-densiity = 0.05

  ;; handles dropping 0.25 with noise

  ;; didn't check capacity








  )












;; --------------------
;; Unit Tests:
;; --------------------


(comment
  (let [reference-implementation
        (fn [{:keys [weights last-activations activations
                     hebbian-plasticity-beta]}]
          (doseq [j (torch/nonzero last-activations)
                  i (torch/nonzero activations)]
            (py/set-item!
             weights
             [j i]
             (* (py.. (py/get-item weights [j i]) item)
                (+ 1 hebbian-plasticity-beta))))
          weights)]
    (filter (comp false? last)
            (for [n (range 100)]
              (let [N (inc (rand-int 10))
                    last-activations (torch/ge (torch/rand [N])
                                               0.5)
                    activations (torch/ge (torch/rand [N]) 0.5)
                    weights (torch/rand [N N] :dtype torch/float)
                    weights-a (py.. weights clone)
                    weights-b (py.. weights clone)]
                [last-activations activations
                 ;; (reference-implementation
                 ;;   {:activations activations
                 ;;    :hebbian-plasticity-beta 0.1
                 ;;    :last-activations last-activations
                 ;;    :weights (py.. weights clone)})
                 ;; (hebbian-plasticity
                 ;;   {:activations activations
                 ;;    :hebbian-plasticity-beta 0.1
                 ;;    :last-activations last-activations
                 ;;    :weights (py.. weights clone)})
                 (torch/allclose
                  (reference-implementation
                   {:activations activations
                    :hebbian-plasticity-beta 0.1
                    :last-activations last-activations
                    :weights weights-a})
                  (hebbian-plasticity
                   {:activations activations
                    :hebbian-plasticity-beta 0.1
                    :last-activations last-activations
                    :weights weights-b}))])))))








;; Lit:

;; Buzsáki G, Mizuseki K. The log-dynamic brain: how skewed distributions affect network operations. Nat Rev Neurosci. 2014 Apr;15(4):264-78. doi: 10.1038/nrn3687. Epub 2014 Feb 26. PMID: 24569488; PMCID: PMC4051294.
