(ns sparser
  (:require [bennischwerdtner.hd.binary-sparse-segmented :as
             hd]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
            [tech.v3.parallel.for :as pfor]
            [tech.v3.datatype.argops :as dtype-argops]
            [tech.v3.datatype.functional :as f]
            [clojure.math.combinatorics :as combo]
            [bennischwerdtner.sdm.sdm :as sdm]))


;;
;; https://youtu.be/NgMWHEC2A4g?si=wzQ_tmUx879Oxsuj
;;
;; Kanerva talks about the properties of a sparse vector with 20 random bits.
;; (here segmented)
;;

;; I can use 20 segments that are 500 wide  (N = 10.000)

;; (/ (long 1e4) 20)
;; 500
;; They are all 500 wide.

(alter-var-root #'hd/default-opts
                (constantly
                 (let [dimensions (long 1e4)
                       segment-count 20]
                   {:bsdc-seg/N dimensions
                    :bsdc-seg/segment-count segment-count
                    :bsdc-seg/segment-length
                    (/ dimensions segment-count)})))


(let [lut (atom {})]
  ;; "encountering a symbol" since symbol and value are
  ;; interchangeable in hdc (Kanerva 2009), why not
  ;; simply call it `prototype`
  ;;
  (defn ->prototype
    "This also stores the symbol in content addressable memeory.

  [[known]] will return the cleaned up symbol.
  "
    [sym]
    (or (@lut sym)
        (let [v (hd/->hv) _ (swap! lut assoc sym v)] v)))
  (defn cleanup-lookup-verbose
    ([query-v] (cleanup-lookup-verbose query-v 0.09))
    ([query-v threshold]
     (->> (map (fn [[k v]]
                 {:k k
                  :similarity (hd/similarity v query-v)
                  :v v})
            @lut)
          (filter (comp #(<= threshold %) :similarity))
          (sort-by :similarity (fn [a b] (compare b a))))))
  (defn cleanup-lookup-value
    [query-v]
    (some->> (cleanup-lookup-verbose query-v)
             first
             :k))
  (defn cleanup-mem [] @lut))
