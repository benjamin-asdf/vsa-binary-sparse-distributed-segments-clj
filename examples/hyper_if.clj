(ns hyper-if
  (:require
   [bennischwerdtner.hd.binary-sparse-segmented :as
    hd]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [tech.v3.parallel.for :as pfor]
   [tech.v3.datatype.argops :as dtype-argops]
   [tech.v3.datatype.functional :as f]))

(let [lut (atom {})]
  ;; "encountering a symbol"
  ;; since symbol and value are interchangeable in hdc (Kanerva 2009), why not simply call it `prototype`
  ;;
  (defn ->prototype
    [sym]
    (or (@lut sym)
        (let [v (hd/->hv) _ (swap! lut assoc sym v)]
          v)))
  (defn cleanup-lookup-verbose
    ([query-v] (cleanup-lookup-verbose query-v 0.1))
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
             :k)))

;; Idea 1:
;;
;; A hyper if
;; In high dimensional computing, the outcome of a calculation could represent
;; the combination of all 'possible' outcomes
;;
;; Interesting here to note is that 'what is possible?' is defined by the threshold, too
;;
;; We can imagine dynamically lowering and increasing the threshold
;; (would model something like 'fast' and 'slow' thinking perhaps)
;;

(defn condition->branches [condition]
  ;; everything above threshold comes out of the thing
  (map :k (cleanup-lookup-verbose condition)))

(defmacro hyper-if
  [condition consequence alternative]
  `(let [condition# ~condition
         branches# (condition->branches condition#)]
     (hd/thin
      (apply
       hd/bundle
       (for [branch# branches#]
         ;; using clojure truthiness of your values now
         (if branch# ~consequence ~alternative))))))

(def both-true-and-false
  (hd/thin
   (hd/bundle
    (->prototype true)
    (->prototype false))))

(defn coin
  []
  (hyper-if both-true-and-false
            (->prototype :heads)
            (->prototype :tails)))

;; all the bookeeping can go away ofc
(map :k (cleanup-lookup-verbose (coin)))

;; => (:heads :tails)
