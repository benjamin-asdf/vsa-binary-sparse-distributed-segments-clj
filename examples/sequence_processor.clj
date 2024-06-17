;;
;; just experiments
;;

(ns sequence-perocessor
  (:require
   [bennischwerdtner.hd.binary-sparse-segmented :as hd]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [tech.v3.parallel.for :as pfor]
   [tech.v3.datatype.argops :as dtype-argops]
   [tech.v3.datatype.functional :as f]))

;; quick content addressable memory
(defprotocol ContentAddressableMemory
  (lookup [this query-v])
  (lookup* [this query-v])
  (store [this v]))

(defn cam-lookup
  [m query-v]
  (let [similarities
          (into [] (pmap #(hd/similarity % query-v) m))]
    (when (seq similarities)
      (let [argmax (dtype-argops/argmax similarities)]
        (when (<= 0.09 (similarities argmax)) (m argmax))))))

(defn cam-lookup*
  [m query-v]
  (let [similarities
          (into [] (pmap #(hd/similarity % query-v) m))]
    (map m
      (map first
        (filter (comp #(< 0.09 %) second)
          (map-indexed vector similarities))))))


(defn cam-store [m v]
  (assert (hd/hv? v))
  (conj m v))

;; there is literature on how to make this smarter,
;; in particular in a `sparse distributed memory`, you don't grow the memory with every new item
;;
(defn ->cam
  []
  (let [m (atom [])]
    (reify
      ContentAddressableMemory
      (lookup [this query-v] (cam-lookup @m query-v))
      (lookup* [this query-v] (cam-lookup* @m query-v))
      (store [this v] (swap! m cam-store v) this))))

(def cam (->cam))

(defn known [x]
  (lookup cam x))

(defn remember-soft
  [x]
  (when-not (known x) (store cam x))
  x)

(defn remember [x] (store cam x) x)


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
          (remember-soft v)
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
             :k)))

(defn cleanup* [query-v]
  (map :k (cleanup-lookup-verbose query-v)))

(defn mix [a b]
  (hd/thin (hd/bundle (->prototype a) (->prototype b))))

(defn ->record [kvps]
  (hd/thin
   (apply
    hd/bundle
    (for [[k v] kvps]
      (hd/bind k v)))))


(comment
  (let [a (hd/->hv)
        b (hd/->hv)
        ab (hd/thin (hd/bundle a b))
        cam [a b ab]]
    (= a (cam-lookup cam a))))


(comment
  (known (remember (->prototype :a)))
  (known (hd/->hv)))

(comment
  (do (store cam (->prototype :a))
      (store cam (->prototype :b))
      (store cam (->prototype :c))
      (= (->prototype :a)
         (lookup cam
                 (hd/thin (hd/bundle (->prototype :a)
                                     (hd/->hv)
                                     (hd/->hv)
                                     (hd/->hv))))))  true)

(defn sequence-marker-1 [k] (hd/->hv))

(def sequence-marker (memoize sequence-marker-1))

(defn ->sequence
  [xs]
  (run! remember xs)
  (hd/thin (apply hd/bundle
             (map-indexed (fn [i x]
                            (hd/bind x (sequence-marker i)))
                          xs))))

(defn h-seq? [exp]
  (and
   (hd/hv? exp)
   (cleanup-cam (h-nth exp 0))))

(defn clj->vsa
  [obj]
  (cond (map? obj) (->record (map (fn [[k v]] [(clj->vsa k)
                                               (clj->vsa
                                                 v)])
                               obj))
        (or (list? obj) (vector? obj)) (->sequence
                                         (map clj->vsa obj))
        (hd/hv? obj) obj
        :else (->prototype obj)))

;; theseq is basically a set where the keys correspond to indices


;; retrieving is the same as with a record
(defn h-nth [hsx idx]
  (hd/unbind hsx (sequence-marker idx)))

(defn cleanup-cam [x] (lookup cam x))

(defn unroll
  [hxs]
  (take-while
   identity
   (map
    cleanup-cam
    (map #(h-nth hxs %) (range)))))

(comment

  (def theseq (->sequence
               [(->prototype :a)
                (->prototype :b)
                (->prototype :c)]))

  (map
   cleanup-lookup-value
   (unroll (clj->vsa [:a :b :c])))

  (= (clj->vsa [:a :b :c]) theseq)

  (cleanup-lookup-value (h-nth (->sequence (map ->prototype [:a :b :c])) 0))
  (cleanup-lookup-value (h-nth (->sequence (map clj->vsa [:a :b :c])) 0))
  (cleanup-lookup-value (h-nth (->sequence (map clj->vsa [:a :b :c])) 0))
  (cleanup-lookup-value (h-nth (clj->vsa [:a :b :c]) 0))

  (clj->vsa [:x :y :z])
  (cleanup-lookup-value (h-nth (clj->vsa [:a [:x :y :z]]) 0))

  (cleanup-lookup-value (h-nth (clj->vsa [:a [:x :y :z]]) 1))

  (hd/similarity
   (clj->vsa [:x :y :z])
   (h-nth (clj->vsa [:a [:x :y :z]]) 1))

  (cleanup-lookup-value (h-nth (h-nth (clj->vsa [:a [:x :y :z]]) 1) 0))

  (h-seq? (h-nth (clj->vsa [:a [:x :y :z]]) 1))
  (h-seq? (h-nth (clj->vsa [:a [:x :y :z]]) 0))

  (unroll-tree (clj->vsa [:a [:x :y :z]]))

  (h-seq?
   (h-nth
    (clj->vsa
     [:a
      false
      [+ 10 20]
      [+ 10 20]]) 2)))

(defn unroll-tree
  [hsx]
  (map (fn [x]
         (if (h-seq? x)
           (unroll-tree x)
           x))
       (unroll hsx)))

(comment
  (h-eval (clj->vsa [+ 1 2 3]))
  ;; => (6)
  (h-eval (clj->vsa [(mix - +) 1 2 3]))
  ;; => (6 -4)
  (h-eval (clj->vsa [+ 1 2 (mix 3 30)]))
  ;; => (6, 33)
)


;;
;; in hyperlisp, expressions are hypervectors
;;

(declare h-apply)

(defn h-if? [exp]
  (and
   (h-seq? exp)
   (= 'if (cleanup-lookup-value
           (cleanup-cam
            (h-nth exp 0))))))

(declare h-eval)

(defn branches [exp]
  (map :k (cleanup-lookup-verbose exp)))

(defn condition->branches [condition]
  ;; everything above threshold comes out of the thing
  (branches condition))

(defn if-condition [exp]
  (cleanup-cam (h-nth exp 1)))
(defn if-consequence [exp]
  (cleanup-cam (h-nth exp 2)))
(defn if-alternative [exp]
  (cleanup-cam (h-nth exp 3)))

(comment
  (cleanup-lookup-value
   (h-nth
    (clj->vsa
     (into [] '(if p consequence alternative)))
    0))
  (clj->vsa
   {:if :if
    :predicate false
    :consequence
    10
    :alternative
    20}))

(defn h-truthy? [o]
  ;; Alternatively,
  ;; could be 'known?'
  ;;
  (when o true))

(defn eval-if
  [exp]
  (let [branches (condition->branches (if-condition exp))]
    ;; (hd/thin)
    (apply
     hd/bundle
     (for [branch branches]
       (if (h-truthy? branch)
         (h-eval (if-consequence exp))
         (h-eval (if-alternative exp)))))))


(comment
  (cleanup*
   (eval-if (clj->vsa ['if (mix true false) :heads :tails])))

  (cleanup*
   (eval-if (clj->vsa ['if (mix 10 false) :heads :tails])))


  (cleanup-lookup-verbose
   (hd/thin
    (apply
     hd/bundle
     [(if-consequence (clj->vsa ['if (mix 10 false) :heads :tails]))
      (if-alternative (clj->vsa ['if (mix 10 false) :heads :tails]))])))

  (cleanup* (if-alternative (clj->vsa ['if (mix 10 false) :heads :tails])))
  (cleanup* (if-consequence (clj->vsa ['if (mix 10 false) :heads :tails])))


  (cleanup-lookup-verbose
   (hd/thin (apply hd/bundle
                   [(clj->vsa :tails)
                    (clj->vsa :heads)])))

  (cleanup-lookup-verbose
   (if-alternative (clj->vsa ['if (mix 10 false) :heads :tails]))))

(defn h-eval
  [exp]
  (cond
    ;;
    ;; possiblity: I. hyper eval looks up hypervectors
    ;; in the cleanup memeory
    ;;
    ;; possiblity: II. hyper eval ruturns hdv, for an
    ;; hdv
    ;;
    ;;

    (h-if? exp) (eval-if exp)

    (h-seq? exp)
    (h-apply (h-eval (first (unroll exp)))
             (map h-eval (rest (unroll exp))))

    ;; (self-evaluating? exp)
    :else
    exp))

(def primitive-op? ifn?)

(defn h-apply
  [op arguments]
  ;; (hd/thin)
  (apply hd/bundle
    (for [op (branches op)]
      (clj->vsa
       (if (primitive-op? op)
         ;; (+ 1 2 3)
         (apply
          op
          (map cleanup-lookup-value arguments))
         ;; what would a hyper lambda be
         ;; doing?
         #_(defprotocol IHyperLambda
             (body [this])
             (environment [this]))
         nil)))))

(comment

  (h-apply (->prototype +)
           (unroll (clj->vsa (into [] (range 3)))))
  (cleanup-lookup-value
   (h-apply (->prototype +)
            (unroll (clj->vsa (into [] (range 3))))))
  (cleanup-lookup-value
   (h-apply (->prototype +)
            (unroll (clj->vsa (into [] (range 3))))))
  3
  (cleanup* (h-apply (mix + -) (unroll (clj->vsa [1 2 3]))))
  ;; (-4 6)
  (cleanup* (h-eval (mix 10 20)))
  (cleanup* (h-eval (clj->vsa ['if (mix 10 20) 30
                               :bananas])))
  (cleanup* (h-eval (clj->vsa ['if (mix 10 false) 30 :bananas])))
  ;; (:bananas 30)

  )


(comment
  (cleanup* (h-eval (clj->vsa [+ 20 5])))
  (25)
  (cleanup*
   (h-eval (clj->vsa [(fn [a b]
                        (+ (inc a) b)) 20 5])))
  (26)


  (cleanup*
   (h-eval (clj->vsa [(mix
                       (fn [a b] (+ (inc a) b))
                       (fn [a b] (+ a a)))
                      20 5])))
  (40 26)


  (def f1 (fn [a b] (+ (inc a) b)))
  (def f2 (fn [a b] (+ a a)))

  (hd/similarity (mix f1 f2) (->prototype f1))

  (hd/similarity
   (clj->vsa [:a :b :c])
   (clj->vsa [:a :b :c]))

  ;; this code is similar...

  (hd/similarity
   (clj->vsa ['if :b :c])
   (clj->vsa [:a :b :c]))

  (let [a (clj->vsa ['if :b :c])
        b (clj->vsa [:x :y :z])]
    (cleanup*
     (hd/bind a (hd/inverse (sequence-marker 0))))))



#_(def pour
  (λ [{:keys [inside]} other-container]
     (substitue inside other-container)))


;; dirty pour

(def pour1
  (fn [container1 container2]
    (hd/thin
     (hd/bundle
      container2
      (hd/bind
       (clj->vsa :inside)
       (hd/unbind container1 (clj->vsa :inside)))))))


(let [my-lava-bucket (pour1 (clj->vsa {:inside :lava})
                            (clj->vsa {:bucket? true
                                       :inside :empty}))]
  {:bucket? (cleanup* (hd/unbind my-lava-bucket
                                 (clj->vsa :bucket?)))
   :inside (cleanup* (hd/unbind my-lava-bucket
                                (clj->vsa :inside)))})

;; => {:bucket? (true) :inside (:lava :empty)}

;; substitution pour
(def pour2
  (fn [container1 container2]
    (let [container2-inside
          (known
           (hd/unbind container2
                      (clj->vsa :inside)))]
      (hd/thin (hd/bundle
                ;; turns out you can dissoc from a sumset by substracting with the kvp
                ;; You might want to do this with cleaned up vecs, or decide some messy ness is good.
                ;;
                (f/- container2
                     (hd/bind (clj->vsa :inside)
                              container2-inside))
                (hd/bind (clj->vsa :inside)
                         (hd/unbind
                          container1
                          (clj->vsa
                           :inside))))))))


(let [my-lava-bucket (pour2 (clj->vsa {:inside :lava})
                            (clj->vsa {:bucket? true
                                       :inside :empty}))]
  {:bucket? (cleanup* (hd/unbind my-lava-bucket (clj->vsa :bucket?)))
   :inside (cleanup-lookup-verbose (hd/unbind my-lava-bucket
                                              (clj->vsa :inside)))})




#_{:bucket? (true)
   :inside ({:k :lava
             :similarity 0.76
             :v #tech.v3.tensor<int8> [10000]
             [0 0 0 ... 0 0 0]})}


;; that's a bucket with both lava and water
(let [lava-bucket (clj->vsa {:bucket? true :inside :lava})
      water-bucket (clj->vsa {:bucket? true :inside :water})
      super-bucket (hd/thin (hd/bundle lava-bucket
                                       water-bucket))]
  {:bucket? (cleanup* (hd/unbind super-bucket
                                 (clj->vsa :bucket?)))
   :inside (cleanup* (hd/unbind super-bucket
                                (clj->vsa :inside)))})


;; {:bucket? (true), :inside (:lava :water)}


;; dirty subst
(def pour3
  (fn [container1 container2]
    (let [container2-inside (hd/unbind container2
                                       (clj->vsa :inside))]
      (hd/thin (hd/bundle (f/- container2
                               (hd/bind (clj->vsa :inside)
                                        container2-inside))
                          (hd/bind (clj->vsa :inside)
                                   (hd/unbind
                                    container1
                                    (clj->vsa
                                     :inside))))))))

(let [lava-bucket (clj->vsa {:bucket? true :inside :lava})
      water-bucket (clj->vsa {:bucket? true :inside :water})
      super-bucket (hd/thin (hd/bundle lava-bucket
                                       water-bucket))
      super-bucket
      (pour3 lava-bucket super-bucket)]

  {:bucket? (cleanup* (hd/unbind super-bucket
                                 (clj->vsa :bucket?)))
   :inside (cleanup* (hd/unbind super-bucket
                                (clj->vsa :inside)))})

;; {:bucket? (true), :inside (:lava)}

(def inside (fn [a] (hd/unbind a (clj->vsa :inside))))

;; just an idea

(defn fulfills-role? [filler role]
  (known
   (hd/bind role (hd/permute filler))))

(defn assign-role [filler role]
  (remember-soft (hd/bind role (hd/permute filler))))

(def spread
  (fn [bread-like butter-like]
    (if-not (fulfills-role? butter-like (clj->vsa :butter))
      ;; 'nonsense'
      (hd/->hv)
      ;; that's just a conj
      (hd/thin (hd/bundle bread-like
                          (hd/bind (clj->vsa :butter)
                                   butter-like))))))


;; it's vegan butter btw
(let [butter-prototype (clj->vsa :butter)
      _ (assign-role (clj->vsa :nectar) (clj->vsa :butter))
      _ (assign-role (clj->vsa :lava) (clj->vsa :butter))]
  (cleanup*
   (hd/unbind
    (spread
     (clj->vsa {:bread? true})
     (clj->vsa :nectar))
    (clj->vsa :butter))))
;; (:nectar)

(let [butter-prototype (clj->vsa :butter)
      _ (assign-role (clj->vsa :nectar) (clj->vsa :butter))
      _ (assign-role (clj->vsa :lava) (clj->vsa :butter))]
  (cleanup*
   (hd/unbind
    (spread
     (clj->vsa {:bread? true})
     (clj->vsa :rocks))
    (clj->vsa :butter))))
;; ()


(let [butter-prototype (clj->vsa :butter)
      _ (assign-role (clj->vsa :nectar) (clj->vsa :butter))
      _ (assign-role (clj->vsa :lava) (clj->vsa :butter))]
  (cleanup*
   (hd/unbind
    (spread
     (clj->vsa {:bread? true})
     (mix :nectar :lava))
    (clj->vsa :butter))))

;; the superposition of nectar and lava happen to be butter-like
;; (because it finds one of them to fulfill the role)

;; (:lava :nectar)

(let [butter-prototype (clj->vsa :butter)
      _ (assign-role (clj->vsa :nectar) (clj->vsa :butter))
      _ (assign-role (clj->vsa :lava) (clj->vsa :butter))]
  (cleanup*
   (hd/unbind
    (spread
     (clj->vsa {:bread? true})
     (mix :nectar :rocks))
    (clj->vsa :butter))))

;; ... rocks nectar would also work
;; (:rocks :nectar)
;;
;; now I imagine nectar with little rocks on the bread.
;; it's crunchy
;;
;;

;; ... unless maybe you select whatever fits the butter role out of the superposition?
;; (but I leave it)



(let [butter-prototype (clj->vsa :butter)
      _ (assign-role (clj->vsa :nectar) (clj->vsa :butter))
      _ (assign-role (clj->vsa :lava) (clj->vsa :butter))]

  ;; (cleanup*
  ;;  (hd/unbind
  ;;   (spread
  ;;    (clj->vsa {:bread? true})
  ;;    (mix :nectar :rocks))
  ;;   (clj->vsa :butter)))


  (map
   cleanup*
   (extract-with-role
    (mix :nectar :rocks)
    (clj->vsa :butter))))



(def spread2
  (fn [bread-like butter-like]


    (if-not (fulfills-role? butter-like (clj->vsa :butter))
      ;; 'nonsense'
      (hd/->hv)
      ;; that's just a conj
      (hd/thin (hd/bundle bread-like
                          (hd/bind (clj->vsa :butter)
                                   butter-like))))))



;; lava and water are not merely associated
;; perhaps they should be *the same*, given the right context
