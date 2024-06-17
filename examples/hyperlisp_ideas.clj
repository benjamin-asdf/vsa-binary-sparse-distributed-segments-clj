(ns hyperlisp-ideas
  (:require
   [bennischwerdtner.hd.binary-sparse-segmented :as
    hd]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [tech.v3.parallel.for :as pfor]
   [tech.v3.datatype.argops :as dtype-argops]
   [tech.v3.datatype.functional :as f]))

;; Make a quick book keeping implementation:

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

;; quick content addressable memory
(defprotocol ContentAddressableMemory
  (lookup [this query-v])
  (store [this v]))

(defn cam-lookup
  [m query-v]
  (let [similarities (into [] (pmap #(hd/similarity % query-v) m))
        argmax (dtype-argops/argmax similarities)]
    (when (<= 0.1 (similarities argmax)) (m argmax))))

(defn cam-store [m v] (conj m v))


;; there is literature on how to make this smarter,
;; in particular in a sparse distributed memory, you don't grow the memory with every new item
(defn ->cam
  []
  (let [m (atom [])]
    (reify
      ContentAddressableMemory
        (lookup [this query-v] (cam-lookup @m query-v))
      (store [this v] (swap! m cam-store v) this))))

(def cam (->cam))

(comment
  (do (store cam (->prototype :a))
      (store cam (->prototype :b))
      (store cam (->prototype :c))
      (= (->prototype :a)
         (lookup cam
                 (hd/thin (hd/bundle (->prototype :a)
                                     (hd/->hv)
                                     (hd/->hv)
                                     (hd/->hv))))))
  true)



;; Make a heteroassociative memory as bridge to clj
;;
;;
;; Then
;;
;; (lookup cam a)
;; -> a
;;
;; (lookup am a)
;; -> clj-data
;;
;; next(a):
;; (lookup cam (unbind a (permute a))
;;

(def h-memory (atom {}))

(defn ->record [kvps]
  (hd/thin
   (apply
    hd/bundle
    (for [[k v] kvps]
      (hd/bind
       (->prototype k)
       (->prototype v))))))




;;  ??
;; Idea 2:
;;
;; Hyper lambda

;; when it encounters args,
;; it allocates prototypes
(defn ->λ
  [f]
  (fn [& args]
    (let [prototypes (map ->prototype args)]
      (hd/thin (apply hd/bundle (f prototypes))))))
