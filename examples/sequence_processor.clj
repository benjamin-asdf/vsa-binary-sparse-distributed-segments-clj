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
   [tech.v3.datatype.functional :as f]
   [clojure.math.combinatorics :as combo]))

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

(defn mix1 [a b]
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
        auto-a-memory [a b ab]]
    (= a (auto-associative-lookup auto-a-memory a))))


(comment
  (known (remember (->prototype :a)))
  (known (hd/->hv)))

(comment
  (do (store auto-a-memory (->prototype :a))
      (store auto-a-memory (->prototype :b))
      (store auto-a-memory (->prototype :c))
      (= (->prototype :a)
         (lookup auto-a-memory
                 (hd/thin (hd/bundle (->prototype :a)
                                     (hd/->hv)
                                     (hd/->hv)
                                     (hd/->hv))))))  true)

(defn sequence-marker-1 [k] (hd/->hv))

(def sequence-marker (memoize sequence-marker-1))

(defn ->sequence
  [xs]
  ;; doesn't allow making lists of noisy sutff ig. Was
  ;; just an attempt... But in principle showcases that
  ;; you can represent sequences with random markers as
  ;; bind
  ;;
  (run!
   (fn [x]
     (when-not
         (known x 0.9)
         (remember x)))
   xs)
  (hd/thin
   (apply hd/bundle
          (map-indexed
           (fn [i x]
             (hd/bind x (sequence-marker i)))
           xs))))

;; theseq is basically a set where the keys correspond to indices
;; retrieving is the same as with a record

(defn h-nth [hsx idx]
  (hd/unbind hsx (sequence-marker idx)))

(defn h-seq? [exp]
  ;; can also be `:nothing`
  ;; doesn't count as seq here atm
  (and
   (hd/hv? exp)
   (known (h-nth exp 0))))

(defn clj->vsa
  [obj]
  (cond (map? obj)
        (->record
         (map
          (fn [[k v]]
            [(clj->vsa k)
             (clj->vsa v)])
          obj))

        (or
         (list? obj)
         (vector? obj))

        ;; you need to make a decision about
        ;; how to deal with the empty sequence

        (if
            (empty? obj)
            (->prototype :nothing)

            (->sequence (map clj->vsa obj)))

        (hd/hv? obj) obj
        ;; there would be alternative ways to do this
        ;; (symbol? obj) (clj->vsa {:symbol
        ;; (->prototype obj)})
        ;; (symbol? obj) (->prototype obj)

        :else (->prototype obj)))

(defn unroll
  [hxs]
  (take-while
   identity
   (map known (map #(h-nth hxs %) (range)))))

(defn unroll-tree
  [hsx]
  (map (fn [x]
         (if (h-seq? x)
           (unroll-tree x)
           x))
       (unroll hsx)))

(defn walk-cleanup
  [form]
  (letfn
    [(f [e]
       (if (hd/hv? e) (cleanup-lookup-value e) (map f e)))]
    (map f form)))



;;
;; A - Ambiguity primitives
;;
;;
;;

(defn mix
  ([a b & args] (hd/thin (apply f/+ a b args)))
  ([a b] (hd/thin (hd/bundle a b))))

(def possibly mix)

(comment
  (hd/similarity (mix (->prototype :a)
                      (->prototype :b)
                      (->prototype :c)
                      (->prototype :d))
                 (->prototype :d))
  0.29)

(def neither (fn [a b] (hd/bind a b)))

(def roughly
  (fn [a amount-of-a] (hd/weaken a (- 1 amount-of-a))))

(defn mostly
  ([a b] (mostly a b 0.3))
  ([a b amount-of-b]
   (hd/thin (hd/bundle a (roughly b amount-of-b)))))

(defn never [e b]
  (hd/thin (f/- e b)))

(def impossibly never)

(comment
  (hd/similarity (impossibly (mix (->prototype :a)
                                  (->prototype :b)
                                  (->prototype :c)
                                  (->prototype :d))
                             (->prototype :d))
                 (->prototype :d))
  0.0


  (let [coin (mix (->prototype :heads) (->prototype :tails))]
    [(cleanup* coin)
     (cleanup* (never coin (->prototype :tails)))])

  ;; [(:heads :tails) (:heads)]

  (let [coin
        (mostly
         (->prototype :heads)
         (->prototype :tails) 0.05)]
    [(cleanup* coin)])
  [(:heads)]


  ;; now if you use a higher threshold for cleanup:
  ;; => it would be quite interesting to modify the threshold dynamically
  ;;
  (let [coin
        (mostly
         (->prototype :heads)
         (->prototype :tails) 0.3)]
    [(hd/similarity coin (->prototype :heads))
     (hd/similarity coin (->prototype :tails))
     [(cleanup* coin)
      (cleanup* coin 0.2)
      (cleanup* (impossibly coin (->prototype :tails)))]])

  ;; [0.87 0.14 [(:heads :tails) (:heads) (:heads)]]



  )


;; a.k.a. a and b's N-space circles (with width threshold) overlap
;;
;; ... or there is a point c in the memory between the 2?
;;
;; (defn necessarily [a b threshold])


;;
;; B - prototypes
;;

(defn non-sense [] (hd/->hv))

;; I think there is something deep about the concept that
;; non-sense and gensym are the same operation
(def create non-sense)





(defn make-hyper [op] (with-meta op {:hyper-fn true}))
(defn mark-hyper [v] (alter-var-root v make-hyper))

(do
  (mark-hyper #'mix)
  (mark-hyper #'possibly)
  (mark-hyper #'neither)
  (mark-hyper #'roughly)
  (mark-hyper #'mostly)
  (mark-hyper #'never)
  (mark-hyper #'impossibly)
  (mark-hyper #'non-sense)
  (mark-hyper #'create))


(comment
  (meta (first (cleanup* (clj->vsa mix))))

  (primitive-type
   (first
    (branches
     (clj->vsa mix))))

  (cleanup*
   (h-eval
    (clj->vsa [mix :a :b])
    (create)))
  '(:b :a)

  )





;;
;; C - analogies
;;


(comment

  (hd/similarity
   (clj->vsa [:x :y :z])
   (h-nth (clj->vsa [:a [:x :y :z]]) 1))
  0.45

  (walk-cleanup
   (unroll-tree (clj->vsa [:a [:x :y :z]])))
  ;; (:a (:x :y :z))

  )



;;
;; I
;;
;; ================
;; The Expression
;; ================
;;
;; In hyperlisp, expressions are hypervectors
;;

;;
;; the *h-environment* could be a sparse distributed memory ?
;;
;; here, I will make the h-enviroment be hypervector map
;; (a set of key value bound pairs like `->record`)
;;
;;
;;
(def ^:dynamic *h-environment* nil)

(declare h-apply)

(declare h-eval)

(defn start-symbol
  [exp]
  (and (h-seq? exp)
       (cleanup-lookup-value (known (h-nth exp 0)))))

(def special? '#{if let lambda})

(defn let? [exp] (= 'let (start-symbol exp)))
(defn lambda? [exp] (= 'lambda (start-symbol exp)))
(defn if? [exp] (= 'if (start-symbol exp)))

(defn augment-environment
  "Returns a new environment with a binding for k->v added."
  [env k v]
  (hd/bundle env (hd/bind k v)))

(defn eval-let
  ([exp] (eval-let exp *h-environment*))
  ([exp env]
   (let [bindings
         (for [[k v]
               (partition
                2
                (unroll (known (h-nth exp 1))))]
           [(known k)
            (h-eval v env)])
         body (known (h-nth exp 2))
         new-env
         (if bindings
           (hd/thin
            (reduce
             (fn [env [k v]]
               (augment-environment env k v))
             env
             bindings))
           env)]
     (def theletenv new-env)
     (h-eval body new-env))))

(comment
  ;; let makes an environment, the evaluator looks up what is bound
  (cleanup*
   (h-eval
    (clj->vsa ['let ['a 100 'b 200] 'a])
    (non-sense)))

  ;; mix primitives work of course
  (cleanup*
   (h-eval
    (clj->vsa ['let ['a [mix 50 20] 'b 200] 'a])
    (create)))
  ;; (20 50)


  ;; and here is something thought provoking:

  (cleanup*
   (h-eval
    (clj->vsa
     ['let ['b 5]
      ['let
       ['b 200]
       'b]])
    (create)))
  ;; (200 5)

  ;; instead of shadowing, the enivronment creates a superposition of values

  ;; and the ambiguity primitives work as expected:

  (cleanup*
   (h-eval
    (clj->vsa
     ['let ['b 5]
      ['let
       ['b 200]
       [never 'b 200]]])
    (create)))
  ;; (5)


  (cleanup*
   (h-eval
    (clj->vsa ['let ['b 5] ['lambda [] 'b]])
    (create)))

  ;; this is a lambda that captures the environment


  (def thelambda (h-eval (clj->vsa ['let ['b 5] ['lambda [] 'b]]) (create)))
  ;; hyperlambdas are hypervectors
  thelambda
  ;; #tech.v3.tensor<int8>[10000]
  ;; [0 0 0 ... 0 0 0]


  ;; hyperlambdas have a body, environment and parameters

  (cleanup* (procedure->body thelambda))
  ;; (b)

  (procedure->environment thelambda)
  ;;   #tech.v3.tensor<int8>[10000]
  ;; [0 0 0 ... 0 0 0]

  ;; the env is just a hypervector representing a map (hypermap ?)
  (cleanup*
   (hd/unbind
    (procedure->environment thelambda)
    (clj->vsa 'b)))
  ;; (5)

  ;; calling a hyperlambda:
  (cleanup* (h-eval (clj->vsa [thelambda]) (create)))
  ;; (5)


  ;; messing with the environment:

  ;; Lambda objects only take the environment into account
  ;; that they are created with
  ;; (no binding of dynamic vars)
  (cleanup*
   (h-eval
    (clj->vsa
     ['let
      ['b 100]
      [thelambda]])
    (create)))
  ;; (5)

  ;; lol, creating hyperlambda with superposition of environments and calling it:

  (cleanup*
   (h-eval
    (clj->vsa
     [['let ['b 5]
       ['let ['b 100]
        ['lambda [] 'b]]]])
    (create)))

  ;; (100 5)

  ;; and now b + 100 means 2 things:

  (cleanup*
   (h-eval
    (clj->vsa
     [['let ['b 5]
       ['let ['b 100]
        ['lambda []
         [+ 100 'b]]]]])
    (create)))
  ;; (200 105)

  ;; redefine coin:

  (def coin-hyper
    (h-eval
     (clj->vsa
      ['let ['coin [mix :heads :tails]]
       ['lambda [] 'coin]])
     (create)))

  ;; this is a hypervector
  coin-hyper
  ;;   #tech.v3.tensor<int8>[10000]
  ;; [0 0 0 ... 0 0 0]

  ;; and the evaluator can call it
  ;; resulting in a hypervector

  (h-eval (clj->vsa [coin-hyper]) (create))

  ;;   #tech.v3.tensor<int8>[10000]
  ;; [0 0 0 ... 0 0 0]

  ;; .. that represents the superposition of multiple symbols:

  (cleanup* (h-eval (clj->vsa [coin-hyper]) (create)))
  ;; (:heads :tails)


  ;; never heads:
  (cleanup*
   (h-eval
    (clj->vsa
     [never [coin-hyper] :heads])
    (create)))
  ;; (:tails)


  ;; this results in value that is possibly heads or tails or foo
  (cleanup*
   (h-eval
    (clj->vsa [possibly [coin-hyper] :foo])
    (create)))

  ;; (:foo :heads :tails)


  ;; a mix of lambdas is also a thing:

  (cleanup*
   (h-eval
    (clj->vsa
     [[mix
       ['lambda ['a] [* 2 'a]]
       ['lambda ['a] [* 10 'a]]]
      10])
    (create)))
  ;; (20 100)

  (cleanup*
   (h-eval
    (clj->vsa
     [[mix
       ['lambda ['a] [* 2 'a]]
       ['lambda ['a] [[mix + - *] 10 'a]]]
      10])
    (create)))


  ;; (20 0 100)


  (cleanup*
   (h-eval
    (clj->vsa
     [[mix
       ['lambda ['a] [* 2 'a]]
       ['lambda ['a] [[mix + - *] 5 'a]]]
      10])
    (create)))

  ;; (20 50 15 -5)

  (cleanup*
   (h-eval
    (clj->vsa
     [[mix + - *] 10 10])
    (create)))

  ;; (100 20 0)


  )

(defn lambda-expr->parameters [exp] (known (h-nth exp 1)))
(defn lambda-expr->body [exp] (known (h-nth exp 2)))

;; Idea 1:
;;
;; eval lambda returns a function, capturing the `environment`
;;
;; Idea 2:
;; eval lambda returns a hypervector
;; that is a record of {:env :parameters :body}
;;
;; -> There is somehow the notion here that 2 hyperlambdas become similar,
;; when their parameters are similar.
;; I feel like there is something we can observe in cognition perhaps.
;; That we find the overlaps between the roles of templates/frames/transformations.
;; E.g. the role of honey on bread, there is something about this honey that is similar
;; to the role of lava on stone.
;;
;;

#_(defn eval-lambda
  [exp environment]
  (let [parameters (lambda-expr->parameters exp)
        body (lambda-expr->body exp)]
    (with-meta
      (fn [& arguments]
        (let [new-env
              (hd/thin
               (reduce (fn [env [k v]]
                         (augment-environment env k v))
                       environment
                       (map vector parameters arguments)))]
          (binding [*h-environment* new-env]
            (h-eval body))))
      {:hyper-fn true})))

(defn eval-lambda
  [exp environment]
  ;; I need the env in the memory, else it get's to
  ;; dirty for what I want to do
  (remember-soft environment 0.9)
  (let [parameters (lambda-expr->parameters exp)
        body (lambda-expr->body exp)
        lambda (clj->vsa {:body body
                          :compound-procedure? true
                          :environment environment
                          :parameters parameters})]
    (remember-soft lambda 0.9)
    lambda))

;; reference:
;; https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/full-text/book/book-Z-H-26.html#%25_idx_4236
;;
(defn extend-environment
  [variables values environment]
  (hd/thin
   (reduce
    (fn [env [k v]] (augment-environment env k v))
    environment
    (map vector variables values))))

(defn procedure->body [proc] (known (hd/unbind proc (clj->vsa :body))))

(defn procedure->parameters [proc]
  (map known (unroll (known (hd/unbind proc (clj->vsa :parameters))))))

(defn procedure->environment [proc]
  (known
   (hd/unbind proc (clj->vsa :environment))))

(defn eval-compound-procedure
  [proc arguments]
  (let [newEnv (extend-environment
                (procedure->parameters proc)
                arguments
                ;; this can be nil, then every thing
                ;; is free variables
                (or (procedure->environment proc)
                    (non-sense)))]
    (h-eval (procedure->body proc) newEnv)))


(comment


  (cleanup-lookup-value
   (lookup-variable
    (clj->vsa 'a)
    (fabrikate-environment
     {'a 10
      'b 100})))
  10


  (cleanup*
   (h-eval
    (clj->vsa [['lambda ['a 'b] [+ 'a 'b]] 20 21])
    (non-sense)))

  ;; (41)

  ;; here is something strange:
  ;; wouldn't it be sick to substitute + for -  in an expression?

  ;; the address of the + is...
  ;; -> nth 0 -> :body -> nth 0
  ;; '((lambda (a b) ([clojure.core/+] a b)) 20 21)

  ;; 1. if + would be part of the lambda env, then substitutiing would be easier

  (cleanup* (h-eval (clj->vsa [['let ['a 100 'b 200]
                                ['lambda ['a 'b] [+ 'a 'b]]]])
                    (non-sense)))
  ;; (300)


  (cleanup*
   (h-eval
    (clj->vsa ['let ['a 100 'b 200] 'a])
    (non-sense)))
  ;; (100)







  (walk-cleanp
   ;; calling known is required, the stuff you get out of
   ;; unbinding from the seq is too dirty
   (unroll (known (h-nth (clj->vsa ['let ['a 100 'b 200] ['lambda ['a 'b] [+ 'a 'b]]]) 1))))
  ;; (a 100 b 200)


  )








(defn variable?
  [exp]
  (symbol? (cleanup-lookup-value exp)))

(defn lookup-variable [exp env]
  (def exp exp)
  (def env env)
  ;; (cleanup* exp)
  ;; (hd/similarity (clj->vsa 'b) exp)
  ;; (cleanup*  (hd/unbind env exp))
  (hd/unbind env exp))

(defn fabrikate-environment
  [kvps]
  (hd/thin
   (reduce (fn [env [k v]]
             (augment-environment env
                                  (clj->vsa k)
                                  (clj->vsa v)))
           (hd/->hv)
           kvps)))




(comment

  (cleanup*
   (hd/permute-inverse
    (h-nth (clj->vsa '(+ a b)) 1)))

  (cleanup-lookup-value
   (hd/permute-inverse (h-nth (clj->vsa '(+ a b)) 0)))


  (cleanup*
   (h-eval
    (clj->vsa
     '(let [a 10 b 20]
        (+ a b)))))

  (let? (clj->vsa '(let [a 10 b 20] (+ a b))))



  (h-nth (clj->vsa '(let [a 10 b 20] (+ a b))) 1)

  (for [[k v]
        (partition
         2
         (unroll
          (known
           (h-nth
            (clj->vsa
             '(let
                  [a 10 b 20]
                  (+ a b)))
            1))))]
    [(cleanup* (hd/permute-inverse k))
     (cleanup* v)])


  ;; ([(a) (10)] [(b) (20)])

  ;; (nil 10 nil 20)

  (cleanup*
   (hd/unbind
    (clj->vsa {:a 10 :b 20})
    (->prototype :a)))

  (cleanup*
   (lookup-variable
    (clj->vsa 'b)
    (->
     (augment-environment
      (hd/->hv)
      (clj->vsa 'a)
      (clj->vsa 10))
     (augment-environment
      (clj->vsa 'b)
      (clj->vsa 20)))))
  (20)

  (= (hd/permute-inverse (clj->vsa 'a)) (->prototype 'a))

  (= (clj->vsa 'a) (hd/permute (->prototype 'a)))

  (cleanup*
   (lookup-variable
    ;; (non-sense)
    (clj->vsa '{a 10 b 20}))))


(comment
  (h-eval (clj->vsa [['let ['a 100 'b 200]
                      ['lambda ['a 'b] [+ 'a 'b]]]]))
  (h-nth (clj->vsa [['let ['a 100 'b 200]
                     ['lambda ['a 'b] [+ 'a 'b]]]])
         0)
  (= (clj->vsa [['let ['a 100 'b 200]
                 ['lambda ['a 'b] [+ 'a 'b]]]])
     (let [e (clj->vsa ['let ['a 100 'b 200]
                        ['lambda ['a 'b] [+ 'a 'b]]])]
       (clj->vsa [e])))
  (cleanup* (h-eval (let [e (clj->vsa ['let ['a 100 'b 200]
                                       ['lambda ['a 'b]
                                        [+ 'a 'b]]])]
                      (clj->vsa [e]))))
  (cleanup* (h-eval (clj->vsa [lambda-with-env])))



  (cleanup*
   (let
       [p

        (h-eval
         (h-nth
          (clj->vsa
           [['let ['a 100 'b 200]
             ['lambda ['a 'b]
              [+ 'a 'b]]]])
          0))
        ]
     ;; (eval-compound-procedure p nil)

       (procedure->environment p)))



  (cleanup*
   (h-eval (clj->vsa [+ 1 2])))

  (cleanup*
   (h-eval (clj->vsa
            [+ 1
             [+ 2 2]])))




  (let
      [p (h-eval
          (h-nth (clj->vsa [['let ['a 100 'b 200]
                             ['lambda ['a 'b]
                              [+ 'a 'b]]]])
                 0))]
    ;; (cleanup*
    ;;    (lookup-variable
    ;;     (clj->vsa 'a)
    ;;     (procedure->environment p)))
      (cleanup* (eval-compound-procedure p nil)))






  (= newEnv env)

  (cleanup* (lookup-variable (clj->vsa 'a) newEnv))

  (hd/similarity
   (clj->vsa 100)
   (lookup-variable (clj->vsa 'a) newEnv))



  )





;; II
;;
;; ======================
;; Multi expressions and branches
;; ======================
;;
;; In hyperlisp,
;;
;; - expressions can evaluate to more than 1 thing (multi symbols)
;; - the `if` expression returns a superposition of the outcome of branches
;; - the `apply` returns a superposition of possible argument lists
;;
;;

(defn branches [exp]
  (cleanup* exp))

(defn condition->branches [condition]
  ;; everything above threshold comes out of the thing
  (branches condition))

(defn if-condition [exp]
  (known (h-nth exp 1)))
(defn if-consequence [exp]
  (known (h-nth exp 2)))
(defn if-alternative [exp]
  (known (h-nth exp 3)))


(defn h-truthy? [o]
  ;; Alternatively,
  ;; could be 'known?'
  ;;
  (when o true))

(defn eval-if
  ([exp] (eval-if exp *h-environment*))
  ([exp env]
   (let [branches (condition->branches (if-condition exp))]
     ;;
     ;; to thin or not to thin is a question
     ;; Because you lose precision
     ;;
     (hd/thin
      (apply hd/bundle
             (for [branch branches]
               (if (h-truthy? branch)
                 (h-eval (if-consequence exp) env)
                 (h-eval (if-alternative exp) env))))))))


(comment
  (cleanup* (eval-if (clj->vsa ['h-if (mix1 true false) :heads :tails])))
  (:tails :heads)
  (cleanup* (eval-if (clj->vsa ['h-if (mix1 10 false) :heads
                                :tails])))
  (:tails :heads)
  )

(defn h-eval
  ;; ([exp] (h-eval exp (or *h-environment* (hd/->hv))))
  [exp env]
  (cond
    ;;
    ;; possiblity: I. hyper eval looks up
    ;; hypervectors in the cleanup memeory
    ;;
    ;; possiblity: II. hyper eval ruturns hdv, for
    ;; an hdv
    ;;
    ;;
    (lambda? exp) (eval-lambda exp env)
    (if? exp) (eval-if exp env)
    (let? exp) (eval-let exp env)
    (h-seq? exp)
    (let [lst (unroll exp)]
      (h-apply
       (h-eval (first lst) env)
       (into [] (map #(h-eval % env) (rest lst)))
       env))
    (variable? exp) (lookup-variable exp env)
    ;; (self-evaluating? exp)
    :else exp))

(def primitive-op? ifn?)

(defn compound-procedure?
  [op]
  (boolean (known (hd/unbind op (clj->vsa :compound-procedure?)))))

(defn primitive-type
  [op]
  (cond (:hyper-fn (meta op)) :hyper-fn
        (ifn? op) :primitive))

;;
;; A cartesian-product arg-branches implementations
;; Different versions are thinkable
;;
(defn arg-branches [arguments]
  (let [arglists (map cleanup* arguments)]
    (apply combo/cartesian-product arglists)))

(defn h-apply
  ([op arguments] (h-apply op arguments *h-environment*))
  ([op arguments env]
   (let [primitive-outcomes
           (for [op (branches op)]
             (clj->vsa (case (primitive-type op)
                         :primitive
                           ;; (+ 1 2 3)
                           (hd/thin
                             (apply hd/bundle
                               ;; (+ (mix1 1 10) 20)
                               (let [branches (arg-branches
                                                arguments)]
                                 (if (seq? branches)
                                   (for [branch branches]
                                     (clj->vsa (apply op
                                                 branch)))
                                   [(clj->vsa (op))]))))
                         :hyper-fn (apply op arguments))))
         compound-outcomes
           (doall (map #(eval-compound-procedure %
                                                 arguments)
                    (filter compound-procedure?
                      (lookup* auto-a-memory op 0.3))))]
     (def procs
       (filter compound-procedure?
         (lookup* auto-a-memory op 0.3)))
     (if-not (seq (concat primitive-outcomes
                          compound-outcomes))
       ;; guess that's an error
       (non-sense)
       (hd/thin (apply hd/bundle
                  (concat primitive-outcomes
                          compound-outcomes)))))))


;; III. The hyperlambda
;; λ
;;
;; hypervectors in hypervector out
;;
;;

;; binding the env means
;;
;; making a superposition of `a` in the environment?
;;
(comment
  (hlet
   [a :banana b :apple]
   (mix1 a b))

  (hlambda
   [{:keys [a b]}]
   (h+ a b)))



(comment
  (h-eval
   [hlambda [{:keys [a b]}] (both a b)])

  (λ [a b] (both a b))

  (h-lambda [a b] (both a b))

  (let [a (->prototype :foo)
        b (->prototype :bar)]
    (h-eval
     (clj->vsa [(hyper-fn [a b] (both a b))]))))



(comment
  (cleanup* (h-apply (->prototype +)
                     (unroll (clj->vsa (into [] (range 3))))))
  ;; (3)

  3
  (cleanup* (h-apply (mix1 + -) (unroll (clj->vsa [1 2 3]))))
  ;; (6 -4)
  ;; (-4 6)
  (cleanup* (h-eval (mix1 10 20)))
  ;; (10 20)

  (cleanup* (h-eval (clj->vsa ['h-if true 30 :bananas])))
  ;; (30)

  (cleanup* (h-eval (clj->vsa ['h-if (mix1 10 20) 30 :bananas])))
  ;; (30)

  (cleanup* (h-eval (clj->vsa ['h-if (mix1 10 false) 30 :bananas])))
  ;; (:bananas 30)
  )


(comment
  (cleanup* (h-eval (clj->vsa [+ 20 5])))
  ;; (25)
  (cleanup*
   (h-eval (clj->vsa [(fn [a b]
                        (+ (inc a) b)) 20 5])))
  ;; (26)


  (cleanup*
   (h-eval (clj->vsa [(mix1
                       (fn [a b] (+ (inc a) b))
                       (fn [a b] (+ a a)))
                      20 5])))
  ;; (40 26)


  ;; to drive this home:

  (def both-dec-and-inc (mix1 dec inc))

  (cleanup* (h-eval (clj->vsa [both-dec-and-inc 7])))
  ;; (6 8)

  (cleanup* (h-eval (clj->vsa [both-dec-and-inc (mix1 5 7)])))
  ;; (8 4 6)

  (cleanup* (h-eval (clj->vsa [both-dec-and-inc (mix1 10 20)])))
  ;; (21 19 11 9)


  (def f1 (fn [a b] (+ (inc a) b)))
  (def f2 (fn [a b] (+ a a)))

  (hd/similarity (mix1 f1 f2) (->prototype f1))

  (hd/similarity
   (clj->vsa [:a :b :c])
   (clj->vsa [:a :b :c]))

  ;; this code is similar...

  (hd/similarity
   (clj->vsa ['h-if :b :c])
   (clj->vsa [:a :b :c]))

  (let [a (clj->vsa ['h-if :b :c])
        b (clj->vsa [:x :y :z])]
    (cleanup*
     (hd/bind a (hd/inverse (sequence-marker 0))))))


(comment
  (h-eval (clj->vsa [+ 1 2 3]))
  ;; => (6)
  (h-eval (clj->vsa [(mix1 - +) 1 2 3]))
  ;; => (6 -4)
  (h-eval (clj->vsa [+ 1 2 (mix1 3 30)]))
  ;; => (6, 33)
  )


(comment



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


  ;; {:bucket? (true)
  ;;  :inside
  ;;  ({:k :lava
  ;;    :similarity 0.76
  ;;    :v #tech.v3.tensor<int8> [10000]
  ;;    [0 0 0 ... 0 0 0]})}


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
       (mix1 :nectar :lava))
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
       (mix1 :nectar :rocks))
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



  ;; --------------------------------
  ;; lava and water are not merely associated
  ;; perhaps they should be *the same*, given the right context
  )





(comment
  (let [a (hd/->hv)
        b (hd/->hv)]
    ;; then it is 50:50
    [(hd/similarity a (mostly a b 1.0))
     (hd/similarity b (mostly a b 1.0))])

  (let [a (hd/->hv)
        b (hd/->hv)]
    ;; this is mostly a
    [(hd/similarity a (mostly a b 0.5))
     (hd/similarity b (mostly a b 0.5))])
  [0.79 0.21]

  ;; I guess 0.3 is at the limit of still being similar to b
  (let [a (hd/->hv)
        b (hd/->hv)]
    ;; this is mostly a
    [(hd/similarity a (mostly a b 0.3))
     (hd/similarity b (mostly a b 0.3))])
  [0.77 0.19])
