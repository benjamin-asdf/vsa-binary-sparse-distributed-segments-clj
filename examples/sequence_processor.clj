
;; just experiments

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

;; quick content addressable memory
(defprotocol ContentAddressableMemory
  (lookup [this query-v])
  (store [this v]))

(defn cam-lookup
  [m query-v]
  (let [similarities
        (into []
              (pmap #(hd/similarity % query-v) m))
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
;; in particular in a `sparse distributed memory`, you don't grow the memory with every new item
;;
(defn ->cam
  []
  (let [m (atom [])]
    (reify
      ContentAddressableMemory
      (lookup [this query-v] (cam-lookup @m query-v))
      (store [this v] (swap! m cam-store v) this))))

(def cam (->cam))

(defn known [x]
  (lookup cam x))

(defn remember-soft [x]
  (when-not
      (known x)
    (store cam x)))

(defn remember [x] (store cam x) x)

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

(defn clj->vsa
  [obj]
  (cond (map? obj)
        (->record (map (fn [[k v]] [(clj->vsa k)
                                    (clj->vsa
                                     v)])
                       obj))
        (vector? obj) (->sequence (map clj->vsa obj))
        (hd/hv? obj) obj
        :else (->prototype obj)))

;; theseq is basically a set where the keys correspond to indices

(def theseq (->sequence
             [(->prototype :a)
              (->prototype :b)
              (->prototype :c)]))

(comment

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
      [+ 10 20]]) 2))
  )


;; retrieving is the same as with a record

(comment
  (cleanup-lookup-value
   (hd/unbind theseq (sequence-marker 0)))
  :a)

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

(defn unroll-tree
  [hsx]
  (map (fn [x]
         (if (h-seq? x)
           (unroll-tree x)
           x))
       (unroll hsx)))

(comment
  (into [] (map cleanup-lookup-value (unroll theseq)))
  [:a :b :c])


(h-eval (clj->vsa [+ 1 2 3]))
;; => (6)

(h-eval (clj->vsa [(mix - +) 1 2 3]))
;; => (6 -4)


(h-eval (clj->vsa [+ 1 2 (mix 3 30)]))
;; => (6, 33)


;;
;; in hyperlisp, expressions are hypervectors
;;

(defn self-evaluating? [exp]
  (hd/hv? exp))

(defn h-seq? [exp]
  (and
   (hd/hv? exp)
   (cleanup-cam (h-nth exp 0))))

(comment
  (h-seq? theseq)
  (h-seq? (hd/->hv)))

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
   (if-alternative (clj->vsa ['if (mix 10 false) :heads :tails])))

  )

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
  (apply
   hd/bundle
   (for [op (branches op)]
     (clj->vsa
      (if (primitive-op? op)
        ;; (+ 1 2 3)
        (apply op (map cleanup-lookup-value arguments))
        ;; compound
        nil)))))


(h-apply
 (->prototype +)
 (unroll
  (clj->vsa (into [] (range 3)))))


(cleanup-lookup-value
 (h-apply (->prototype +)
          (unroll (clj->vsa (into [] (range 3))))))


(cleanup-lookup-value
 (h-apply
  (->prototype +)
  (unroll (clj->vsa (into [] (range 3))))))

3

(cleanup* (h-apply (mix + -) (unroll (clj->vsa [1 2 3]))))
;; (-4 6)


(cleanup* (h-eval (mix 10 20)))


(cleanup*
 (h-eval (clj->vsa ['if (mix 10 20) 30 :bananas])))

(cleanup*
 (h-eval (clj->vsa ['if (mix 10 false) 30 :bananas])))

(def exp (clj->vsa ['if (mix 10 20) 30 :bananas]))

(cleanup*
 (h-eval (clj->vsa ['if (mix 10 false) 30 :bananas])))

(cleanup*
 (eval-if (clj->vsa ['if (mix 10 false) 30 :bananas])))



(cleanup*
 (hd/bundle
  (clj->vsa :bananas)
  (hd/->hv)))


(def theexp
  (clj->vsa
   ['if true [+ 10 20] [+ 20 5]]))

(h-seq? (h-nth theexp 2))

(cleanup* (h-nth theexp 0))

(cleanup* (h-nth theexp 1))
(h-seq? (h-nth theexp 2))
(h-eval (h-nth theexp 2))
(known (h-nth (h-nth theexp 2) 0))

(cleanup* (h-eval theexp))

(cleanup* (h-eval (clj->vsa [+ 20 5])))
(25)

(cleanup*
 (h-eval (clj->vsa [(fn [a b]
                      (+ (inc a) b)) 20 5])))
(26)

(cleanup*
 (h-eval (clj->vsa [
                    (mix
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

(hd/similarity
 (clj->vsa ['if :b :c])
 (clj->vsa [:a :b :c]))

(let [a (clj->vsa ['if :b :c])
      b (clj->vsa [:x :y :z])]
  (cleanup*
   (hd/bind a (hd/inverse (sequence-marker 0)))))

(let [container-lava (clj->vsa {:inside :lava})]
  (hd/unbind (->prototype)))
