(ns sequence-perocessor
  (:require
   [bennischwerdtner.hd.binary-sparse-segmented :as hd]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [tech.v3.parallel.for :as pfor]
   [tech.v3.datatype.argops :as dtype-argops]
   [tech.v3.datatype.functional :as f]
   [clojure.math.combinatorics :as combo]))

;; ================ ad hoc memory lib ===========

;; quick associative memory
(defprotocol AssociativeAddressableMemory
  (lookup [this query-v]
    [this query-v threshold])
  (lookup* [this query-v]
    [this query-v threshold])
  (store [this v])
  (mem [this]))

(defn auto-associative-lookup
  ([m query-v] (auto-associative-lookup m query-v 0.09))
  ([m query-v threshold]
   (let [similarities
         (into [] (pmap #(hd/similarity % query-v) m))]
     (when (seq similarities)
       (let [argmax (dtype-argops/argmax similarities)]
         (when (<= threshold (similarities argmax)) (m argmax)))))))

(defn auto-associative-lookup*
  ([m query-v] (auto-associative-lookup* m query-v 0.09))
  ([m query-v threshold]
   (let [similarities
         (into [] (pmap #(hd/similarity % query-v) m))]
     (map m
          (map first
               (filter (comp #(< threshold %) second)
                       (map-indexed vector similarities)))))))

(defn auto-associative-store [m v]
  (assert (hd/hv? v))
  (conj m v))

;; there is literature on how to make this smarter,
;; in particular in a `sparse distributed memory`, you don't grow the memory with every new item
;;
(defn ->auto-a-memory
  []
  (let [m (atom [])]
    (reify
      AssociativeAddressableMemory
      (lookup [this query-v]
        (auto-associative-lookup @m query-v))
      (lookup [this query-v threshold]
        (auto-associative-lookup @m query-v threshold))
      (lookup* [this query-v]
        (auto-associative-lookup* @m query-v))
      (lookup* [this query-v threshold]
        (auto-associative-lookup* @m query-v threshold))
      (store [this v] (swap! m auto-associative-store v) this)
      (mem [this] @m))))

(def auto-a-memory (->auto-a-memory))

(defn known
  "Cleanup x with the autoassociative memory."
  ([x] (known x 0.09))
  ([x threshod]
   (lookup auto-a-memory x threshod)))

(defn remember-soft
  ([x] (remember-soft x 0.9))
  ([x threshod]
   (when-not (known x threshod) (store auto-a-memory x))
   x))

(defn remember [x] (store auto-a-memory x) x)

;; Make a quick book keeping implementation:

(def hyper-symbols-symbols
  ["🐂" "🐛" "🚌" "Ψ" "Ϟ" "🪓" "🌈"])

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
        (let [v (hd/->hv)
              _ (swap! lut assoc sym v)]
          ;; !
          ;; (always a new vec, we just created it)
          (remember v)
          v)))
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

(defn cleanup*
  ([query-v] (cleanup* query-v 0.09))
  ([query-v threshold]
   (map :k (cleanup-lookup-verbose query-v threshold))))


;; ================ ad hoc memory lib ends here ===========


;; this is like making all possible combinations a x 0, a x 1, a x 2, ...

(cleanup* (hd/unbind
           (hd/bind (hd/thin (hd/bundle (->prototype :a)
                                        (->prototype :b)
                                        (->prototype :c)))
                    (hd/thin (hd/bundle (->prototype 0)
                                        (->prototype 1)
                                        (->prototype 2))))
           (->prototype :a)))
;; (2 1 0)
