(ns what-is-the-dollar-in-mexico
  (:require
   [bennischwerdtner.hd.binary-sparse-segmented :as h]))

;; classic:
;; "What is the Dollar of Mexico?" (Kanerva 2010)
;;

;; Make a quick book keeping implementation:

(let [lut (atom {})]
  (defn symbol->hv
    [sym]
    (or (@lut sym)
        (let [v (h/->hv) _ (swap! lut assoc sym v)] v)))
  (defn cleanup-lookup-verbose
    ([query-v] (cleanup-lookup-verbose query-v 0.1))
    ([query-v threshold]
     (->> (map (fn [[k v]]
                 {:k k
                  :similarity (h/similarity v query-v)
                  :v v})
               @lut)
          (filter (comp #(<= threshold %) :similarity))
          (sort-by :similarity (fn [a b] (compare b a))))))
  (defn cleanup-lookup-value
    [query-v]
    (some->> (cleanup-lookup-verbose query-v)
             first
             :k)))


(def mexico-record
  (h/thin
   (h/bundle
    (h/bind (symbol->hv :capital) (symbol->hv 'mxc))
    (h/bind (symbol->hv :currency) (symbol->hv 'peso))
    (h/bind (symbol->hv :name) (symbol->hv 'mex)))))

(def usa-record
  (h/thin (h/bundle (h/bind (symbol->hv :capital)
                            (symbol->hv 'wdc))
                    (h/bind (symbol->hv :currency)
                            (symbol->hv 'dollar))
                    (h/bind (symbol->hv :name)
                            (symbol->hv 'usa)))))


(let [result
      (h/unbind mexico-record
                ;; this represents the query
                (h/unbind usa-record (symbol->hv 'dollar)))]

  ;; I am excited...
  [
   (first (cleanup-lookup-verbose result))

   :result

   (cleanup-lookup-value result)])


;; tada!

#_[{:k peso
    :similarity 0.12
    :v #tech.v3.tensor<int8> [10000]
    [0 0 0 ... 0 0 0]}
   :result
   peso]


;; ----------------------------------------------------------------------------------------
;; We have more `structure`, by exploiting multiple kinds of operations on the vectors.
;; Compare to context word vectors, which only use addition. We see that addition in hdc is an unordered set operation.
;; ----------------------------------------------------------------------------------------


;; ----------------------------------------------------------------------------------------

;; This is cooler when bind is invertible
;; Then you can do this: (omitting details)
;;

(let [mexico-record
      {:capital 'mxc :currency 'peso :name 'mex}
      usa-record
      {:capital 'wdc :currency 'dollar :name 'usa}
      mapping (bind usa-record mexico-record)]
  (bind mapping 'dollar)
  ;; -> peso
  )

;; You can store the entire db in 'mapping', and query it.
;;
;; But since the bind and unbind here aren't inverses, we need to unbind from the USA record
;;

;; -----------------------------
;;
;; II. The difference of variable and value is fuzzy:
;;

(cleanup-lookup-value (h/unbind usa-record (symbol->hv 'dollar)))
:currency

;; The usa representation says :currency when I ask for Dollar.
;;
;; High dimensional computing is structured, yet fuzzy in cool ways.
;; The hyper vectors are a great way to have fun with a computer.
;;
