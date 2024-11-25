(ns iris-sdm
  (:require [tech.v3.datatype.functional :as f]
            [bennischwerdtner.sdm.sdm :as sdm]
            [bennischwerdtner.hd.core :as hd]
            [bennischwerdtner.pyutils :as pyutils]
            [libpython-clj2.python :refer [py. py..] :as py]
            [libpython-clj2.require :refer [require-python]]
            [tech.v3.dataset :as ds]))

(require-python '[torch :as torch])

;; iris classifier with SDM
;; ------------------------------------------------

(def species ["Iris-setosa" "Iris-versicolor" "Iris-virginica"])
(def feature-names ["SepalLengthCm" "SepalWidthCm" "PetalLengthCm" "PetalWidthCm"])

;; -------------------------------

(defn norm
  "Normalize a value to exist between 0 and 1 (inclusive)."
  [val start stop]
  (max 0 (min 1 (/ (- val start) (- stop start)))))

;; -------------------

(defn projection
  [{:keys [high low level]} value]
  (let [num-levels (py.. level (size 0))]
    (py/get-item level
                 (long (Math/floor
                         (* (dec num-levels)
                            (norm value low high)))))))

;; ---------------------------------------------------

(declare num-levels)
(declare feature-symbol-seeds)
(declare calibration)

(defn calibrate
  [inputs]
  (let [info (update-vals (select-keys (group-by :col-name
                                                 (ds/brief
                                                  inputs))
                                       feature-names)
                          peek)]
    (into {}
          (for [feat feature-names]
            (let [level (hd/level num-levels)
                  high (-> info
                           (get feat)
                           ;; :max
                           :quartile-3)
                  low (-> info
                          (get feat)
                          ;; :min
                          :quartile-1)]
              [feat {:high high :level level :low low}])))))


;; hyperdim record:
;; feature -> value
;;
;; the 'percept', I want to say 'Gestalt tag' (Greg Egan)
;;
(defn percept
  [dat]
  (apply hd/superposition
         (map (fn [feat feat-symbol]
                (hd/bind feat-symbol
                         (projection (calibration feat)
                                     (dat feat))))
              feature-names
              feature-symbol-seeds))
  ;; (hd/thin)
  )

;; ----------------------------------
;;
;; Explore phase:
;;
;; 1. 'encounter' in the 'world':
;;
;; percept -> species (from species-seeds)
;;
;; 2. Put into sdm:
;;
;; percept ->  species
;; key         value

;; -----------------------------------------

;; Usage:
;;
;; 1. Given dat,
;; 2. build percept
;; 3. query SDM,
;; 4. cleanup with species-seeds
;;

;; ------------------------------

(defn cleanup
  ([mem x] (cleanup mem x 0.18))
  ([mem x threshold]
   (let [scores (hd/similarity mem x)
         [value index] (into [] (torch/max scores :dim -1))]
     (when (<= threshold (py.. value item))
       (torch/index_select mem -2 index)))))

(defn cleanup-verbose
  ([mem x] (cleanup-verbose mem x 0.18))
  ([mem x threshold]
   (let [scores (hd/similarity mem x)
         [value index] (into [] (torch/max scores :dim -1))]
     (when (<= threshold (py.. value item))
       {:idx index
        :item (torch/index_select mem -2 index)
        :sim value}))))

;; ---------------------------------------------

(declare attention-mask)
(declare species->seed)
(declare decoder-threshold)
(declare species-seeds)

(defn train!
  [sdm train-dat]
  (doseq [dat train-dat]
    (let [address (percept dat)
          address (torch/bitwise_and attention-mask address)
          value (-> dat
                    (get "Species")
                    species->seed)]
      (sdm/write sdm address value decoder-threshold)))
  sdm)

(defn classify
  [sdm dat]
  (let [result (:result (sdm/lookup
                         sdm
                         (torch/bitwise_and attention-mask (percept dat))
                         ;; (percept dat)
                         1
                         decoder-threshold))]
    (when result
      (species (py.. (:idx (cleanup-verbose species-seeds
                                            result))
                 item)))))

(comment
  (let [count-failures
        (for [n (range 20)]
          (count
           (do
             (def decoder-threshold 2)
             (def test-split 0.9)
             (def num-levels 10)
             (def iris-count
               (count (ds/rows
                       (ds/->dataset
                        "/home/benj/tmp/Iris.csv"))))
             (def train-count (long (* test-split iris-count)))
             (def test-count (- iris-count train-count))
             (def feature-symbol-seeds
               (hd/seed (count feature-names)))
             (def species-seeds (hd/seed (count species)))
             (def species->seed
               (into {}
                     (map vector species species-seeds)))
             (def iris-shuffled
               (into []
                     (shuffle
                      (ds/rows
                       (ds/->dataset
                        "/home/benj/tmp/Iris.csv")))))
             (def iris-train
               (take train-count iris-shuffled))
             (def iris-test
               (take-last test-count iris-shuffled))
             (def calibration (calibrate iris-train))
             ;; seems like I got better outcome with some attention
             ;;
             ;; the attention mask is a kind of context dependent thinning,
             ;; dropping the indices that are shared between all inputs.
             ;; The idea is to amplify the differences and remove the overlaps
             (def attention-threshold-count 100)
             (def attention-mask
               (py..
                   (let [all-percepts
                         (torch/stack
                          (into []
                                (for [dat iris-train]
                                  (percept dat))))
                         threshold-count
                         attention-threshold-count]
                     (let [superpos (hd/superposition
                                     all-percepts)
                           total-size (py.. superpos
                                        (size 0))
                           [top-values top-indices]
                           (into []
                                 (torch/topk
                                  superpos
                                  threshold-count))
                           mask (torch/ones [total-size])]
                       (torch/index_put
                        mask
                        [top-indices]
                        (torch/zeros [threshold-count]))))
                   (to :dtype torch/int8
                       :device pyutils/*torch-device*)))
             (def model
               (train! (sdm/->sdm
                        {:address-count (long 1e6)
                         :address-density 0.0003
                         :word-length (long 1e4)})
                       iris-train))
             ;; dropping random items from the
             ;; content matrix doesn't change the
             ;; outcome much
             ;; this does something similar overall to the
             ;; attention-mask, but indiscrimetely
             ;; could try drop the top content,
             ;; but that should be very similar
             ;; (sdm/decay model 0.2)
             (into []
                   (filter (comp false? #(nth % 2))
                           (for [dat iris-test]
                             (let [c (classify model dat)]
                               [dat c
                                (= (get dat "Species")
                                   c)])))))))]
    (- 1 (f// (f/mean count-failures) (count iris-test))))
  0.9133333333333333)























;; ------------------------------
(comment
  (let [by-species (ds/group-by-column (ds/->dataset
                                        iris-train)
                                       "Species")]
    (for [node by-species]
      (py.. (torch/nonzero
             (torch/ge
              (torch/sum
               (torch/stack
                (into []
                      (map
                       (comp
                        ;; (fn [x]
                        ;;   (torch/bitwise_and
                        ;;    attention-mask
                        ;;    x))
                        percept)
                       (ds/rows (second node)))))
               :dim
               0)
              40))
        (size))))

  ;; (torch.Size([23, 1]) torch.Size([50, 1])
  ;; torch.Size([26, 1]))
  )
















;; ------------------------------------------------------------

;; Lit:
;; 1.
;; Robust Clustering using Hyperdimensional Computing
;; 2312.02407v1
;; 2.
;; https://iris.rais.is/en/publications/classification-and-feature-extraction-of-hyperdimensional-data-us/fingerprints/?sortBy=alphabetically
