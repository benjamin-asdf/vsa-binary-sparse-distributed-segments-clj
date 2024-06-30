;;
;; just experiments
;;

(ns sequence-processor-sdm
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

(defn ->record
  [kvps]
  (hd/thin (apply hd/bundle
             (for [[k v] kvps] (hd/bind k v)))))

(comment
  (let [a (hd/->hv)
        b (hd/->hv)
        ab (hd/thin (hd/bundle a b))
        auto-a-memory [a b ab]]
    (= a (auto-associative-lookup auto-a-memory a))))

(comment
  (known (remember (->prototype :a)))
  (known (hd/->hv)))


(defn sequence-marker-1 [k] (hd/->hv))

(def sequence-marker (memoize sequence-marker-1))

(defn ->sequence
  [& xs]
  ;; doesn't allow making lists of noisy sutff ig. Was
  ;; just an attempt... But in principle showcases that
  ;; you can represent sequences with random markers as
  ;; bind (that is equivalent to a random projection
  ;; for each position in this implementation)
  ;;
  (run! (fn [x] (when-not (known x 0.9) (remember x))) xs)
  (hd/thin (apply hd/bundle
             (map-indexed (fn [i x]
                            (hd/bind x (sequence-marker i)))
                          xs))))

;; seq is basically a set where the keys correspond to indices
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
  (cond (map? obj) (->record (map (fn [[k v]] [(clj->vsa k)
                                               (clj->vsa
                                                 v)])
                               obj))
        (or
         (list? obj)
         (vector? obj))
          ;; you need to make a decision about
          ;; how to deal with the empty sequence
          (if (empty? obj)
            (->prototype :nothing)
            (apply ->sequence (map clj->vsa obj)))
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
  ([a] a)
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
     (cleanup* (never coin (->prototype :tails)))]

    ;; [(:heads :tails) (:heads)]

    (let [coin
          (mostly
           (->prototype :heads)
           (->prototype :tails) 0.05)]
      [(cleanup* coin)]))
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

(defn non-sense [] (hd/->hv))

;; I think there is something deep about the concept that
;; non-sense and gensym are the same operation
(def create non-sense)

(defn make-hyper [op] (with-meta op {:hyper-fn true}))
(defn mark-hyper [v]
  (alter-meta! v assoc :hyper-fn true)
  (alter-var-root v make-hyper))

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

;;
;; A II - prototypes
;;
;;
;;


;;
;; B - The means of combination
;;
;;

(defn h-first [hsx]
  (hd/unbind hsx (sequence-marker 0)))

;; basically substitute the keys in the record with n - 1
(defn h-rest [hsx]
  (let
      ;; that's what it boils down to anyway I think
      [r (rest (unroll hsx))]
    (apply ->sequence r)))

(defn pair [a b]
  (->sequence a b))

(do
  (mark-hyper #'h-first)
  (mark-hyper #'h-rest)
  (mark-hyper #'pair)
  (mark-hyper #'->sequence)
  (mark-hyper #'unroll))


;; the primitives of key-value pairs

(def bind hd/bind)
(def inverse hd/inverse)
(def unbind hd/unbind)
(def release unbind)

(defn ->struct [kvps])

(do
  (mark-hyper #'bind)
  (mark-hyper #'inverse)
  (mark-hyper #'unbind)
  (mark-hyper #'release)
  (mark-hyper #'h-first)
  (mark-hyper #'h-rest)
  (mark-hyper #'pair))


(comment
  (walk-cleanup (unroll (h-rest (pair (->prototype :a) (->prototype :b)))))
  ;; (:b)

  (cleanup* (h-first (h-rest (pair (->prototype :a) (->prototype :b)))))
  ;; (:b)

  )


;;
;; C - analogies / templates / frames
;;
;; the means of abstraction
;;
;; wip ...
;;

(defn substitute [e k v])

;;
;; I
;;
;; ================
;; The Expression
;; ================
;;
;; In hyperlisp, expressions are hypervectors
;;
;; The evaluator
;; -----------------------------------
;;
;; exp is a symbol: lookup in the environment
;;
;; exp is if: Evaluate the condition, lookup the condition in the clj memeory,
;;            for each truthy branch, evaluate the consequence
;;            fore each falsy branch, evaluate the alternative
;;         evaluate to the superposition of the outcomes
;;
;; exp is let: Evaluate the bindings, augment the environment, evaluate the body
;;
;; exp is lambda: Return a hypervector that represents the lambda
;;
;; exp is a sequence: Evaluate the first element and treat it as a function
;;                    Evaluate the rest of the elements and treat them as arguments
;;
;;
;;             if the operator is a primitive, apply the primitive, with the clj values from cleanup memory
;;             if the operator is hyper-fn, do not cleanup to clj, else the same
;;
;;                Do this with all 'argument branches' (cartisian product of possible arguments in this implementation)
;;
;;             if the operator is a lambda, evaluate the lambda
;;
;;               To eval a lambda:
;;                 augment the environment with the parameters and arguments
;;                 evaluate the body with the new environment
;;
;;             The result is the superposition of the outcomes
;;
;;
;;
;; if the exp is anything else, it evaluates to itself
;;
;;
;;


;;
;; the *h-environment* could be a sparse distributed memory ?
;;
;; here, I will make the h-enviroment be hypervector map
;; (a set of key value bound pairs like `->record`)
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
(defn branch? [exp] (= 'branch (start-symbol exp)))

(defn augment-environment
  "Returns a new environment with a binding for k->v added."
  [env k v]
  (hd/bundle env (hd/bind k v)))

(defn eval-let
  ([exp] (eval-let exp *h-environment*))
  ([exp env]
   (let [bindings (for [[k v] (partition
                                2
                                (unroll (known (h-nth exp
                                                      1))))]
                    [(known k) (h-eval v env)])
         body (known (h-nth exp 2))
         new-env
           (if bindings
             (hd/thin
               (reduce (fn [env [k v]]
                         (augment-environment env k v))
                 env
                 bindings))
             env)]
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
  (cleanup* (h-eval (clj->vsa [thelambda])))
  ;; (5)


  ;; messing with the environment:

  ;; Lambda objects only take the environment into account
  ;; that they are created with
  ;; (no binding of dynamic vars)
  (cleanup* (h-eval (clj->vsa ['let ['b 100] [thelambda]])))
  ;; (5)

  ;; lol, creating hyperlambda with superposition of environments and calling it:

  (cleanup*
   (h-eval (clj->vsa [['let ['b 5]
                       ['let ['b 100]
                        ['lambda [] 'b]]]])))

  ;; (100 5)

  ;; and now b + 100 means 2 things:

  (cleanup*
   (h-eval (clj->vsa [['let ['b 5]
                       ['let ['b 100]
                        ['lambda [] [+ 100 'b]]]]])))
  ;; (200 105)


  ;; redefine coin:

  (def coin-hyper
    (h-eval (clj->vsa ['let ['coin [mix :heads :tails]]
                       ['lambda [] 'coin]])))

  ;; this is a hypervector
  coin-hyper
  ;;   #tech.v3.tensor<int8>[10000]
  ;; [0 0 0 ... 0 0 0]

  ;; and the evaluator can call it as function,

  (h-eval (clj->vsa [coin-hyper]))

  ;;   #tech.v3.tensor<int8>[10000]
  ;; [0 0 0 ... 0 0 0]

  ;; the ouctome is a hypervector
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
    (clj->vsa [[mix
                ['lambda ['a] [* 2 'a]]
                ['lambda ['a] [* 10 'a]]]
               10])))
  ;; (20 100)

  (cleanup*
   (h-eval
    (clj->vsa
     [[mix ['lambda ['a] [* 2 'a]]
       ['lambda ['a]
        [[mix + - *] 10 'a]]] 10])))


  ;; (20 0 100)

  (cleanup*
   (h-eval
    (clj->vsa
     [[mix ['lambda ['a] [* 2 'a]]
       ['lambda ['a]
        [[mix + - *] 5 'a]]] 10])))

  ;; (20 50 15 -5)

  (cleanup* (h-eval (clj->vsa [[mix + - *] 10 10])))
  (20 100 0)


  ;; (100 20 0)


  (cleanup*
   (h-eval
    (clj->vsa
     ['let ['outcome
            [['lambda []
              ['if [possibly true false]
               :heads :tails]]]]
      [impossibly 'outcome :tails]])))
  (:heads))

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
  (let [new-env (extend-environment
                 (procedure->parameters proc)
                arguments
                ;; this can be nil, then every thing
                ;; is free variables
                (or (procedure->environment proc)
                    (non-sense)))]
    (h-eval (procedure->body proc) new-env)))

(defn variable?
  [exp]
  (symbol? (cleanup-lookup-value exp)))

(defn lookup-variable [exp env]
  (hd/unbind env exp))

(defn fabricate-environment
  [kvps]
  (hd/thin
   (reduce (fn [env [k v]]
             (augment-environment env
                                  (clj->vsa k)
                                  (clj->vsa v)))
           (hd/->hv)
           kvps)))

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
  (if
      (= :nothing o)
      false
      (when o true)))

(defn eval-if
  ([exp] (eval-if exp *h-environment*))
  ([exp env]
   (let [branches (condition->branches
                    (h-eval (if-condition exp) env))]
     ;; to thin or not to thin is a question
     ;; Because you lose precision
     ;;
     (if-not (seq branches)
       ;; that would be an error?
       (non-sense)
       (hd/thin (apply hd/bundle
                  (for [branch branches]
                    (if (h-truthy? branch)
                      (h-eval (if-consequence exp) env)
                      (h-eval (if-alternative exp)
                              env)))))))))

(defn
  branch->antecedent
  [exp]
  (known (h-nth exp 1)))

(defn
  branch->postcedent
  [exp]
  (known (h-nth exp 2)))

;; do you return a sequence of outcomes?
;; or a superposition of outcomes?

(defn eval-branch
  [exp env]
  (let [antecedent (h-eval (branch->antecedent exp) env)
        postcedent (h-eval (branch->postcedent exp))
        collapsed-branches (lookup* auto-a-memory
                                    antecedent
                                    ;; dynamic threshold
                                    ;; would be
                                    ;; interesting
                                    0.1)]
    ;; [collapsed-branches
    ;;  antecedent
    ;;  postcedent]
    (hd/thin (apply hd/bundle
               (for [collapsed collapsed-branches]
                 (h-apply postcedent [collapsed] env))))))

(defn h-eval
  ([exp] (h-eval exp (or *h-environment* (non-sense))))
  ([exp env]
   (cond
     ;;
     ;; possiblity: I. hyper eval looks up
     ;; hypervectors in the cleanup memeory
     ;;
     ;; possiblity: II. hyper eval ruturns hdv, for an
     ;; hdv
     ;;
     ;;
     (branch? exp) (eval-branch exp env)
     (lambda? exp) (eval-lambda exp env)
     (if? exp) (eval-if exp env)
     (let? exp) (eval-let exp env)
     (h-seq? exp)
     (let [lst (unroll exp)]
       (h-apply (h-eval (first lst) env)
                (into [] (map #(h-eval % env) (rest lst)))
                env))
     (variable? exp) (lookup-variable exp env)
     ;; (self-evaluating? exp)
     :else exp)))

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
     (if-not (seq (concat primitive-outcomes
                          compound-outcomes))
       ;; guess that's an error
       (non-sense)
       (hd/thin (apply hd/bundle
                  (concat primitive-outcomes
                          compound-outcomes)))))))

;;
;;
;; III. The reader
;;
;; This is for convinience.
;;
;;
;; - Clojure sets become a sumset (bundle).
;; - Clojure maps become a sumset of bound pairs.
;; - Clojure vectors become a hyper sequence.
;;

(defn set-expr [exp]
  (into [possibly] exp))

(defn map-expr [exp]
  (into
   [mix]
   (for [[k v] exp]
     [bind k v])))

(defn vec-expr [exp]
  (into [->sequence] exp))

(defn analyse-expression
  [clj-exp]
  (cond
    (set? clj-exp) (set-expr (map analyse-expression
                                  clj-exp))
    (map? clj-exp)
    (map-expr (map (fn [[k v]] [(analyse-expression k)
                                (analyse-expression v)])
                   clj-exp))
    (and (list? clj-exp) (= 'let (first clj-exp)))
    (list 'let
          (into []
                (map analyse-expression (nth clj-exp 1)))
          (analyse-expression (nth clj-exp 2)))

    (and (list? clj-exp) (= 'lambda (first clj-exp)))
    (list 'lambda
          (into []
                (map analyse-expression (nth clj-exp 1)))
          (analyse-expression (nth clj-exp 2)))


    (and (list? clj-exp) (= 'fn (first clj-exp)))
    (eval clj-exp)

    (vector? clj-exp) (vec-expr (map analyse-expression
                                     clj-exp))
    (list? clj-exp)
    (into [] (map analyse-expression clj-exp))


    ;; guess I'm kludgin it up, but hey clj meta
    ;; data and namespaces are simply amazing
    :else (let [hypersymbols
                (into {}
                      (filter
                       (fn [[sym v]]
                         (when (var? v)
                           (:hyper-fn (meta v))))
                       (ns-map *ns*)))]
            (or
             (hypersymbols clj-exp)
             clj-exp))))

(defn h-read [clj-exp]
  (clj->vsa (analyse-expression clj-exp)))

(defmacro h-read-code
  [code]
  `(h-read '~code))

;; (alter-meta! #'h-read (constantly {:foo :bar}))
;; (meta #'h-read)
;; (meta (first [h-read]))

(comment
  (analyse-expression '#{a b c})
  ;; [possibly a c b]

  (cleanup*
   (h-eval
    (clj->vsa
     ['let ['a 10 'b 20]
      (analyse-expression '#{a b})])))

  (cleanup*
   (hd/unbind
    (h-eval (clj->vsa (analyse-expression {:a 10 :b 20})))
    (->prototype :a)))
  ;; (10)

  (cleanup*
   (hd/unbind
    (h-eval (h-read {:a 100 :b 'lol}))
    (->prototype :a)))
  ;; (100)


  (cleanup*
   (hd/unbind
    (h-eval
     (h-read-code
      {:a 100 :b lol}))
    (clj->vsa :a)))

  ;; (100)

  (cleanup* (h-eval (h-read-code (let [a 100] a))))
  ;; (100)

  ;; ... is the same:
  (cleanup*
   (h-eval
    (h-read '(let [a 100] a))))

  ;; and it is hyperlisp:

  (cleanup*
   (h-eval
    (h-read-code
     (let [a 100]
       (let [a 200]
         a)))))
  ;; (200 100)

  (cleanup*
   (h-eval
    (h-read-code
     (let [a 100]
       (let [a 200]
         (mix
          (+ a a)
          (* a a)))))))


  ;; .. yay this is hyperlisp code now
  (cleanup* (h-eval (h-read-code (mix :a :b))))
  ;; (:b :a)

  (analyse-expression '(let [a #{10 20} b {:a #{20 100}}]))

  (cleanup*
   (h-eval
    (h-read-code
     (let [a #{10 20}] a))))


  ;; I discovered that in this configuration the elements of the possiblity mix
  ;; become just thresholdy likely
  ;;
  ;; Then, it depends on the environment seed what the outcome is.
  ;; Quite fascinating.
  ;;
  ;; Playing around with thresholds and 'perspectives' comes to mind as one of the next steps.
  ;; Perhaps a mechanism might find a perpective vector that splits 2 concepts and so forth.
  ;;

  (def seed-with-3-outcomes seed)
  (def seed-with-2-outcomes seed)
  (def seed (hd/->seed))

  ;; after I defined the seeds, the outcome is deterministic:

  [(cleanup* (h-eval (h-read-code
                      (let [a #{10 20} b :c] (possibly b a)))
                     seed-with-3-outcomes))
   (cleanup* (h-eval (h-read-code
                      (let [a #{10 20} b :c] (possibly b a)))
                     seed-with-2-outcomes))]

  ;; [(:c 10 20) (:c 20)]


  (walk-cleanup (unroll (h-eval (clj->vsa [->sequence 1 2 3]))))
  (walk-cleanup (unroll (h-eval (clj->vsa (analyse-expression [1 2 3])))))

  (walk-cleanup (unroll (h-eval (h-read-code [1 2 3]))))
  ;; (1 2 3)



  (cleanup*
   (h-eval
    (h-read-code
     (let [a [1 2 3]])))))

(comment
  (cleanup*
   (h-apply
    (->prototype +)
    (unroll
     (clj->vsa (into [] (range 3))))))
  ;; (3)
  3
  (cleanup*
   (h-apply
    (mix1 + -)
    (unroll (clj->vsa [1 2 3]))))
  ;; (6 -4)
  ;; (-4 6)
  (cleanup*
   (h-eval (mix1 10 20)))
  ;; (10 20)
  (cleanup*
   (h-eval
    (clj->vsa
     ['if true 30 :bananas])))
  ;; (30)
  (cleanup*
   (h-eval
    (clj->vsa
     ['if (mix1 10 20) 30 :bananas])))
  ;; (30)
  (cleanup*
   (h-eval
    (clj->vsa
     ['if
      (mix1 10 false)
      30
      :bananas])))
  ;; (30 :bananas)

  (seq? (list 1 2 3))
  (seq? [1 2 3])
  (seqable? [1 2 3])
  (seqable? "fo"))

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

  ;; this code is similar...
  (hd/similarity
   (clj->vsa ['if :b :c])
   (clj->vsa [:a :b :c])))

(comment
  ;; => (6)
  (h-eval
   (clj->vsa [(mix1 - +) 1 2 3]))
  ;; => (6 -4)
  (h-eval
   (clj->vsa [+ 1 2 (mix1 3 30)]))
  ;; => (6, 33)
  (cleanup*
   (h-eval
    (clj->vsa
     ['if [mix true false] :a :b])
    (create)))
  ;; (:b :a)
  )

(comment
  ;; hyperlisps 'collapse' primitive 'branch'
  ;; looks up e in the associative memory,
  ;; branch function is called called for each known hdv
  ;;
  ;; So 'certainty' and branching are 2 sides of the same coin
  ;;
  ;; Or 'measurement'

  (cleanup*
   (hd/unbind
    (h-eval
     (h-read-code
      (branch
       (mix :a :b)
       (fn [it] {:foo it})))
     (hd/->seed))
    (clj->vsa :foo)))

  (cleanup*
   (hd/unbind (h-eval (h-read-code {:foo :bar})
                      (hd/->seed))
              (clj->vsa :foo)))


  ;; ----------------------
  ;; collapse primitives
  ;; ----------------------

  (cleanup*
   (hd/unbind
    (h-eval (h-read-code
             (branch (mix :a :b)
                     (lambda [it] {:foo it})))
            (hd/->seed))
    (clj->vsa :foo)))
  ;; (:b :a)


  ;; maybe apply is sufficient for a branch primitive though:
  (cleanup*
   (h-eval
    (h-read-code
     ((fn [it] it) (mix 10 20)))))
  ;; (20 10)
  ;; .. this does the same thing basically atm

  (cleanup*
   (hd/unbind
    (h-eval
     (h-read-code
      ((fn [it] {:foo it}) (mix :a :b))))
    (clj->vsa :foo)))
  ;; (:a :b)

  )

(comment

  (def spread-butter
    (h-read-code
     (lambda
      (bread butter)
      {:bread bread
       :butter butter})))

  (def output (h-eval (clj->vsa
                       [spread-butter :bread :honey])))

  (cleanup*
   (hd/unbind output (clj->vsa :butter)))
  ;; (:honey)
  )

(comment
  (def seed+water->plant
    '(lambda
      (seed water kind)
      {:plant kind}))

  (do
    (defn ergo [antecedent postcedent]
      (bind antecedent postcedent))
    (mark-hyper #'ergo))

  (h-read-code (ergo (mix :seed :water) :plant))

  (cleanup*
   (hd/unbind
    (h-eval (h-read-code (ergo (mix :seed :water) :plant)))
    (hd/unbind
     (hd/unbind
      (h-eval (h-read-code
               {:seed :seed-a :water :water}))
      (clj->vsa :seed-a))
     (clj->vsa :water))))

  (let [m (h-eval (h-read-code
                   {:a 10 :b 20}))]
    (cleanup*
     (hd/unbind m (hd/bundle
                   (clj->vsa :a)
                   (clj->vsa :b)))))


  (let [m (h-eval (h-read-code
                   (ergo (mix :a :b) :c)))]
    (cleanup*
     (hd/unbind m
                (hd/bundle
                 (clj->vsa :a)
                 (clj->vsa :b)))))
  ;; (:c)

  (cleanup*
   (h-eval
    (h-read-code
     (release
      (ergo (mix :seed :water) :plant)
      (mix :seed :water)))))
  ;; (:plant)

  (cleanup-lookup-verbose
   (h-eval
    (h-read-code
     (release
      (ergo (mix :seed :water) :plant)
      (mix :seed)))))

  (cleanup-lookup-verbose
   (h-eval
    (h-read-code
     (release
      (ergo (mix :seed :water) :plant)
      (mix :seed :water)))))

  (cleanup-lookup-verbose
   (h-eval
    (h-read-code
     (let [water :water]
       (release
        (ergo (mix :seed :water) :plant)
        (mix water :seed)))))))
