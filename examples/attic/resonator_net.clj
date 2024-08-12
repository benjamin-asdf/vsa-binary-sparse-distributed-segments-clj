(ns resonator-net
  (:require
   [bennischwerdtner.pyutils :as pyutils]
   [bennischwerdtner.hd.prot :as prot]
   [tech.v3.datatype.functional :as f]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [tech.v3.datatype.bitmap :as bitmap]
   [fastmath.random :as fm.rand]
   [fastmath.core :as fm]
   [bennischwerdtner.hd.binary-sparse-segmented :as hd]
   [tech.v3.datatype.unary-pred :as unary-pred]
   [tech.v3.datatype.argops :as dtype-argops]
   [bennischwerdtner.hd.data :as hdd]
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python.ffi :as py-ffi]
   [libpython-clj2.python :refer [py. py.. py.-] :as py]))


;; The problem:
;;
;; - with structured data at hand, where multiple bind factors contributte to a composite vector,
;; - finding the contributing vectors is a factorization problem
;; - comes up for instance when looking for the trace of a a tree from data.clj
;; - (really kinda comes up all the the time, the moment we make composits)

;; ---------------------------------------------
;; Given a composit vector s, find the contributing factors (of seed vectors)
;; -------------------------------------------

(def s
  (hdd/clj->vsa* [:*
                  (hdd/clj->vsa [:x 0])
                  (hdd/clj->vsa [:y 0])
                  (hdd/clj->vsa [:z 0])]))

;; You can do this by enumerating all possibilities, but this goes into a combinatorial explosion.
;;
;;

;;
;; all the vectors are from known codebooks, x is from codebook {x1,...,xD} etc.
;;

;;
;; 1. x-estimate, y-estimate, z-estimate, the current estimates
;;    Initialize them as the superposition of all possibble vectors
;;

(def D 100)
(def x-codebook (into #{} (map (fn [i] (hdd/clj->vsa [:x i])) (range D))))
(def y-codebook (into #{} (map (fn [i] (hdd/clj->vsa [:y i])) (range D))))
(def z-codebook (into #{} (map (fn [i] (hdd/clj->vsa [:z i])) (range D))))

;; superposition of all x'ses ,   ∑ x-codebook

;; exploring the properties of this x-initial:

;; this breaks down for a vector of this density:
(->>
 (hdd/cleanup-verbose
  (hdd/clj->vsa* x-codebook))
 (map :k))

(->>
 ;; but threshold 1.0 makes sense of course, the seeds are there 100%
 (hdd/cleanup-verbose
  (hdd/clj->vsa* x-codebook)
  1.0)
 (map :k)
 (map first)
 (into #{}))
'#{:x}
;; only :x came out

(hd/similarity (first x-codebook) (hdd/clj->vsa* x-codebook))
1.0


(let
    [x-estimate (hdd/clj->vsa* x-codebook)
     y-estimate (hdd/clj->vsa* y-codebook)
     z-estimate (hdd/clj->vsa* z-codebook)]
  ;; this would be the crossproduct of the whole thing,
  ;; but I don't have a bind that preservers density
  ;;
  ;;
    (hd/bind* [x-estimate y-estimate z-estimate])

    (hd/similarity
     z-estimate
     (hd/unbind
      (hd/unbind s x-estimate) y-estimate)))



;; a particular factor can be inferred from s based ooon the estimates for the other two
;;

;; but maybe not a problem, if D is small like 7

(do
  (def D 5)
  (def x-codebook (into #{} (map (fn [i] (hdd/clj->vsa [:x i])) (range D))))
  (def y-codebook (into #{} (map (fn [i] (hdd/clj->vsa [:y i])) (range D))))
  (def z-codebook (into #{} (map (fn [i] (hdd/clj->vsa [:z i])) (range D)))))

(let
    [x-estimate (hdd/clj->vsa* x-codebook)
     y-estimate (hdd/clj->vsa* y-codebook)
     z-estimate (hdd/clj->vsa* z-codebook)]
  ;; this would be the crossproduct of the whole thing,
  ;; but I don't have a bind that preservers density
  ;;
  ;;
    (hd/bind* [x-estimate y-estimate z-estimate])

    (hd/similarity
     z-estimate
     (hd/unbind
      (hd/unbind s x-estimate)
      y-estimate)))


(hdd/cleanup*
 (hd/unbind s
            (hd/bind
             (hdd/clj->vsa [:y 0])
             (hdd/clj->vsa [:z 0]))))
'([:x 0])



(hd/unbind s (hd/bind (hdd/clj->vsa* y-codebook) (hdd/clj->vsa* z-codebook)))

(hdd/cleanup-verbose
 (hd/unbind s
            (hd/bind
             (hdd/clj->vsa* y-codebook)
             (hdd/clj->vsa* z-codebook)))
 0.0)



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

(defn preallocated-alphabet
  [n]
  (dtt/->tensor (repeatedly n hd/->seed) :datatype :int8))

(defn preallocated-seed-item-memory
  [pool]
  (let [lut (atom {})
        seed (fn []
               ;; is of course an error if you ask
               ;; for more seeds than pool
               (nth pool (count @lut)))]
    (reify
      prot/ItemMemory
        (prot/m-cleanup [this q]
          (get (clojure.set/map-invert @lut) q))
        (prot/m-clj->vsa [this item]
          (or (get @lut item)
              (let [v (seed)]
                (swap! lut assoc item v)
                v))))))

(def alphabet (preallocated-alphabet 10))
(def seed-mem (preallocated-seed-item-memory alphabet))

(hd/similarity
 (first alphabet)
 (dtt/reduce-axis alphabet f/sum 0))

(defn codebook-cleanup-1
  [codebook-matrix q]
  (torch/select codebook-matrix
                0
                (torch/argmax
                 (torch/mv
                  codebook-matrix
                  (py.. (pyutils/ensure-torch q)
                    (to :dtype torch/float16))))))

(defn codebook-cleanup
  [codebook-matrix q]
  (pyutils/torch->jvm
    (py.. (codebook-cleanup-1 codebook-matrix q)
      (to :dtype torch/int8))))

(defn ->codebook-matrix
  [pool]
  (py.. (pyutils/ensure-torch pool)
    (to :dtype torch/float16)))


(defprotocol ResonatorNetwork
  (lookup [this factor-memories query]))

(defn resonator-decode
  [factor-alphabets query]
  (reductions
    (fn [{:keys [estimates]}]
      {:estimates
         ;; estimates t + 1 this can be done in
         ;; parallel, I would have to develop the torch
         ;; code a bit more to do that.
         (into
           []
           (for [[idx factor-alphabet]
                   (map-indexed vector factor-alphabets)]
             ((:cleanup factor-alphabet)
               (hd/unbind query
                          ;; unbind with everything
                          ;; except *our* factor. In
                          ;; effect, everybody reports
                          ;; what the closest match is,
                          ;; given the context
                          (hd/bind*
                            (select-keys
                              estimates
                              (clojure.set/difference
                                (into #{} (range estimates))
                                #{idx})))))))})
    {:estimates (into []
                      ;; initial estimates, the
                      ;; superposition of each alphabet
                      (map
                       #(dtt/reduce-axis % f/sum 0)
                       :pool
                       factor-alphabets))}))

(def pool (preallocated-alphabet 5))

(def alphabets
  (for [depth (range 5)]
    (let [pool (dtt/map-axis pool hd/permute depth)
          ;; each pool has the unit vector so it's
          ;; factor can come out to "1"
          pool (dtt/->tensor (concat pool
                                     [(hd/unit-vector)]))
          codebook-matrix (->codebook-matrix pool)]
      {:cleanup (fn [x]
                  ;; dt to jvm because I didn't
                  ;; implement the VSA in torch yet
                  (pyutils/torch->jvm
                    (py.. (codebook-cleanup codebook-matrix)
                          (to :dtype torch/int8))))
       :pool pool})))

(def seed-mem (preallocated-seed-item-memory pool))

(binding [hdd/*item-memory* seed-mem]
  (def tree
    (hdd/clj->vsa*
     [:+
      [:* [:*> :left :right :left] :a]])))

(binding [hdd/*item-memory* seed-mem]
  (hdd/cleanup
   (codebook-cleanup
    (->codebook-matrix pool)
    (hdd/clj->vsa*
     [:. tree
      [:*> :left :right :left]]))))
:a

;; {:left :right} (5 total random seeds)

(binding [hdd/*item-memory* seed-mem]
  (hdd/cleanup
   (let [outcome-for-3]
     (codebook-cleanup
      (->codebook-matrix
       (let [depth 2]
         (dtt/map-axis pool #(hd/permute-n % depth))))
      (hd/unbind
       tree
       (hd/bind*
        [(dtt/reduce-axis pool f/sum 0)
         (hd/permute (dtt/reduce-axis pool f/sum 0))]))))))

(binding [hdd/*item-memory* seed-mem]
  (hd/similarity
   (hd/permute-n (hdd/clj->vsa :left) 2)
   (hd/unbind
    tree
    (hd/bind*
     [(dtt/reduce-axis pool f/sum 0)
      (hd/permute (dtt/reduce-axis pool f/sum 0))]))))

[:*> :left :right :left]


(comment

  (def x-reportior (hdd/clj->vsa* #{:left :right}))
  (def y-reportior (hdd/clj->vsa* #{[:> :left 1] [:> :right 1]}))
  (def z-reportior (hdd/clj->vsa* #{[:> :left 2] [:> :right 2]}))

  (def s (hdd/clj->vsa* [:*> :left :right :left]))


  (hd/bind*
   [(hdd/clj->vsa* [:*> :left :right :left])
    (hdd/clj->vsa* :a)])

  (f/sum
   (hdd/clj->vsa* [:*> :left :right :left]))

  (f/sum
   (hd/bind*
    [(hdd/clj->vsa* [:*> :left :right :left])
     (hdd/clj->vsa* :a)]))

  ;;
  ;; the 'problem' here is that the bind essentially thins,
  ;; dropping down the contribution of everything in the sumset
  ;;
  (f/sum (hd/bind* [z-reportior y-reportior]))
  100.0

  ;; the resultant hdv, now has 1/4th of the info for [:left 1], etc.


  (hd/unbind s (hd/bind* [z-reportior y-reportior]))
  ;; ~ x-reportior

  (hd/similarity
   x-reportior
   (hd/unbind s (hd/bind* [z-reportior y-reportior])))
  0.26
  ;; I guess because of some symmetry, we come to 1/4th here, too
  ;; this overlap comes from the 1/4th [:left] contribution
  ;;

  ;;

  (def resonators
    [(->codebook-matrix (dtt/->tensor (hdd/clj->vsa*
                                       [:left :right])))
     (->codebook-matrix (dtt/->tensor (hdd/clj->vsa*
                                       [[:> :left 1]
                                        [:> :right 1]])))
     (->codebook-matrix (dtt/->tensor (hdd/clj->vsa*
                                       [[:> :left 2]
                                        [:> :right 2]])))])
  ;; resonator:

  (->> (last
        (take 5
              (iterate
               (fn [{:keys [estimates]}]
                 {:estimates
                  (into []
                        (map-indexed
                         (fn [idx x-tilde]
                           (codebook-cleanup
                            (resonators idx)
                            (hd/unbind
                             s
                             (hd/bind*
                              (into
                               []
                               (keep
                                identity
                                (map-indexed
                                 (fn [i e]
                                   (when-not
                                       (= i idx)
                                       e))
                                 estimates)))))))
                         estimates))})
               {:estimates [x-reportior y-reportior
                            z-reportior]})))
       :estimates
       (map-indexed (fn [idx e] (hd/permute-n e (- idx))))
       (map hdd/cleanup*))
  '((:left) (:right) (:left))

  y-reportiorx


  (hd/bind* [x-reportior y-reportior z-reportior])

  (let [estimates [x-reportior y-reportior z-reportior]
        idx 0]
    (hd/similarity
     x-reportior
     (hd/unbind
      s
      (hd/bind*
       (into [] (keep identity (map-indexed (fn [i e] (when-not (= i idx) e)) estimates)))))))


  (def depth 5)

  (concat
   (dtt/map-axis
    (dtt/->tensor
     (hdd/clj->vsa* [:left :right]))
    #(hd/permute-n % 1))
   [(hd/unit-vector)])

  (hd/similarity
   (hd/permute-n (hdd/clj->vsa* :left) 1)
   (first
    (dtt/map-axis
     (dtt/->tensor
      (hdd/clj->vsa* [:left :right]))
     #(hd/permute-n % 1))))

  (hdd/cleanup*
   (hd/permute-inverse
    (codebook-cleanup
     (->codebook-matrix
      (dtt/map-axis
       (dtt/->tensor
        (hdd/clj->vsa* [:left :right])
        :datatype :int8)
       #(hd/permute-n % 1)))
     (hdd/clj->vsa* [:> :left]))))
  '(:left)

  (hdd/cleanup*
   (codebook-cleanup
    (->codebook-matrix
     (dtt/map-axis
      (dtt/->tensor
       (hdd/clj->vsa* [:left :right])
       :datatype :int8)
      #(hd/permute-n % 0)))
    (hdd/clj->vsa* :left)))
  '(:left)



  (dtt/ensure-native q)
  (pyutils/ensure-torch q)


  (dtt/tensor->dimensions
   ;; (hdd/clj->vsa* [:> :left])
   (hd/permute-n (hdd/clj->vsa* :left) 1))
  (dtt/simple-dimensions?
   ;; (hdd/clj->vsa* [:> :left])
   (hd/permute-n (hdd/clj->vsa* :left) 1))

  (dtt/ensure-native (hd/permute-n (hdd/clj->vsa* :left) 1))

  (dtt/simple-dimensions?
   ;; (hdd/clj->vsa* [:> :left])
   (hdd/clj->vsa* :left))

  (= s (hd/unbind s (hd/unit-vector)))





  ;; depth 3 is already high

  (def resonators
    (into []
          (for [d (range 3)]
            (let [codebook (dtt/map-axis (dtt/->tensor
                                          (hdd/clj->vsa*
                                           [:left :right
                                            ;; quickly breaks down
                                            ;;
                                            ;; :foo :bar
                                            ;; (hd/unit-vector)
                                            ])
                                          :datatype
                                          :int8)
                                         #(hd/permute-n % d))]
              {:codebook codebook
               :codebook-matrix (->codebook-matrix
                                 codebook)}))))

  (let [s (hdd/clj->vsa* [:*> :right :right :right])]
    (->>
     (last
      (take
       30
       (iterate
        (fn [{:keys [estimates]}]
          {:estimates
           (into
            []
            (map-indexed
             (fn [idx x-tilde]
               (codebook-cleanup
                (:codebook-matrix (resonators idx))
                (do
                  (def thes
                    (hd/unbind
                     s
                     (hd/bind*
                      (into []
                            (keep identity
                                  (map-indexed
                                   (fn [i e]
                                     (when-not
                                         (= i idx)
                                         e))
                                   estimates))))))
                  (hd/unbind
                   s
                   (hd/bind*
                    (into []
                          (keep identity
                                (map-indexed
                                 (fn [i e]
                                   (when-not
                                       (= i idx)
                                       e))
                                 estimates))))))))
             estimates))})
        {:estimates
         (into []
               (map (comp #(dtt/reduce-axis % f/sum 0)
                          :codebook)
                    resonators))})))
     :estimates
     (map-indexed (fn [idx e] (hd/permute-n e (- idx))))
     (map hdd/cleanup)))

  '(:right :right :right))
