;;
;; These are just notes atm
;;

(ns sequence-perocessor
  (:require
   [bennischwerdtner.hd.binary-sparse-segmented :as
    hd]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [tech.v3.parallel.for :as pfor]
   [tech.v3.datatype.argops :as dtype-argops]
   [tech.v3.datatype.functional :as f]))

;; Make a quick book keeping implementation:

(def hyper-symbols-symbols
  ["🐂" "🐛" "🚌" "Ψ" "Ϟ" "🪓" "🌈"])

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

(comment
  (let [a (hd/->hv)
        b (hd/->hv)
        ab (hd/thin (hd/bundle a b))
        cam [a b ab]]
    (= a (cam-lookup cam a))))


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



(def sequence-token (->prototype :sequence))

(defn sequence? [o]
  (lookup cam (hd/bind o sequence-token)))


(defn ->sequence
  [xs]
  ;; version 1: Pointer chains that would be
  ;; heteroassociative chain can start anywhere, but if
  ;; the items comes up twice, it collides
  ;;
  ;;
  ;; A sequence is now the same thing as it's first
  ;; item.
  ;;
  ;; This poses the problem of multiple sequences with
  ;; the same start. Also, sequences with the same item
  ;; at pos n, would not be differentiated
  ;;
  ;; A solution to these problems might be 'context'.
  ;; The caller can implement this by making `a` a
  ;; gensym,. When thy construct the sequence, they can
  ;; bind all vectors with `a`. Then when they consume
  ;; the sequence, they can unbind with `a`.
  ;;
  ;; If the processor is updating the context while it
  ;; unrolls the sequence, it would be able to branch
  ;; depending on the context.
  ;;
  ;;
  ;;
  ;; Q is the identity of the sequence
  ;; Map all elements to Q-space,. This protects the
  ;; memory of different sequences
  (let [Q (hd/->hv)
        xs (concat [Q] xs)
        xs (map (fn [x] (hd/bind x Q)) xs)]
    (doall
     (map (fn [a b]
            (let [a->b (hd/bind a (hd/permute a))]
              ;; [ a . ]
              (store cam a)
              ;; [ . b ]
              (store cam a->b)
              (swap! h-memory assoc a->b b)
              a))
          xs (next xs)))
    Q))



(=
 (->sequence [(->prototype :a) (->prototype :b) (->prototype :c)])
 (->sequence [(->prototype :a) (->prototype :b) (->prototype :c)]))





(let [a (->prototype :a)]
  (= a
     (hd/unbind (hd/bind a (hd/permute a))
                (hd/permute a))))

(comment
  (let [seq-handle (->sequence [(->prototype :a)
                                (->prototype :b)
                                (->prototype :c)])]
    (take-while
     identity
     (iterate (fn [a]
                (when a
                  (let [a (lookup cam a)]
                    (when a
                      (let [a->b (hd/bind a (hd/permute a))
                            b->b (lookup cam a->b)]
                        (@h-memory a->b))))))
              (lookup cam seq-handle)))))




;; returns a potentionally infinite sequence of the next items in the sequence
(defn seq-read
  [seq-identifier]
  (let [first-item (@h-memory seq-identifier)]
    (take-while
      identity
      (iterate
        (fn [a]
          (when a
            (let [a (lookup cam a)]
              (when a
                (let [a->b (hd/bind a (hd/permute a))
                      a->b (lookup cam a->b)]
                  (when a->b
                    (if (sequence? a->b)
                      (seq-read a->b)
                      (when-let [b (@h-memory a->b)]
                        (hd/unbind b seq-identifier)))))))))
        first-item))))


(seq-read
 (->sequence
  [(->prototype :a)
   (->prototype :b)
   (->prototype :c)]))





(map
 cleanup-lookup-value
  (let [Q
        (->sequence [(->prototype :a)
                     (->prototype :b)

                     (->prototype :c)

                     ])
        seq-first (lookup cam (hd/bind Q Q))]
    Q
    (map (fn [x] (hd/unbind x Q))
      (take-while
        identity
        (drop
         1
         (iterate (fn [a]
                    (when a
                      (let [a (lookup cam a)]
                        (when a
                          (let [a->b (hd/bind a
                                              (hd/permute a))
                                b->b (lookup cam a->b)]
                            (@h-memory a->b))))))
                  seq-first))))))





(defn seq-unroll
  [Q]
  (let [seq-first (lookup cam (hd/bind Q Q))]
    (map
     (fn [x] (hd/unbind x Q))
     (take-while
      identity
      (drop
       1
       (iterate
        (fn [a]
          (when a
            (let [a (lookup cam a)]
              (when a
                (let [a->b (hd/bind a (hd/permute a))
                      b->b (lookup cam a->b)
                      b (@h-memory a->b)]
                  (when b
                    (or (let [b (hd/unbind b Q)]
                          (when (lookup cam (hd/bind b b))
                            (seq-unroll b)))
                        b)))))))
        seq-first))))))


(map
 cleanup-lookup-value
 (let [Q (->sequence [(->prototype :a)
                      ;; (->prototype :b)
                      (->sequence [(->prototype :a)
                                   (->prototype :b)
                                   ;; (->prototype :c)
                                   ])
                      ;; (->prototype :c)
                      ])]
   (seq-unroll Q)))





(let [Q (hd/->hv)
      a (->prototype :a)
      b (->prototype :b)
      xs (map (fn [x] (hd/bind Q x)) [a b])
      first-item
      (hd/bind (first xs) (hd/permute (first xs)))
      ]
  (hd/bind a (hd/permute a))
  (hd/similarity
   (hd/bind Q a)
   (first xs)))





;; quickly do the boring thing and make a hdv list processor

(defn clj->vsa
  [obj]
  (cond
    (map? obj) (->record obj)
    (vector? obj)
    (->sequence (map clj->vsa obj))
    :else (->prototype obj)))

(comment
  (into []
        (map
         cleanup-lookup-value
         (seq-next (clj->vsa [:a :b :c]))))
  [:a :b :c]

  (into []
        (map
         cleanup-lookup-value
         (seq-next
          (clj->vsa [:a
                     [:b :d :e :f]
                     :c]))))
  (into []
        (map
         cleanup-lookup-value
         (seq-next
          (clj->vsa
           [:b :d :e :f]))))
  [:b :d :e :f]


  )


(def lambda-token (->prototype :λ))
(def if-token (->prototype :if))

(defn primitive-op? [o]
  (ifn? o))

(defn lambda? [o]
  (= :λ (cleanup-lookup-value o)))

(defn if? [o]
  (= :if (cleanup-lookup-value o)))

(defn sp-eval [hv]
  (cond
    )

  )


(defn sp-apply [op arguments]
  (cond
    (primitive-op? op)
    (apply op arguments)



    ))
