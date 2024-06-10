;; -------------------------
;; Just notes
;; -------------------------
(ns bennischwerdtner.hd.hyperlisp
  (:refer-clojure
   :exclude
   [apply eval symbol? intern true?])
  (:require
   [clojure.core :as clj]
   [bennischwerdtner.hd.binary-sparse-segmented :as hd]))
;; --------------------------------
;; Lambda Calculus With High Dimensional Computing
;; --------------------------------
;;
;; Lisp
;; Recursive Functions of Symbolic Expressions and Their Computation by Machine, Part I
;; http://www-formal.stanford.edu/jmc/recursive.html

;;
;;

(def coin
  (multi (intern true) (intern false)))

(defn flip []
  ;; flip returns a hypervector that is equally
  ;; similar to true and false
  (eval coin))


(defmacro lambda [& sigs]
  ;; try to get the answer from memory
  `(let []
     (clj/ifn ~@sigs)))

(macroexpand
 '(lambda [] (if (flip) "Tails" "Heads")))

(clojure.core/ifn [] (if (flip)))

(eval
 (macroexpand
  '(lambda [] (if (flip)))))

(def coinflip
  (lambda [] (if (flip) "Tails" "Heads")))

@(coinflip)
;; -> ["Tails", "Heads"]



;; resolves to more than 1 thing
(defprotocol IHyperSymbol)

;; a hdv that evals to itself
(defprotocol IHyperAtom)

;;
(defprotocol IHyperLambda)


(defn lambda? [o]
  (satisfies? IHyperLambda o))

;;
(defprotocol IHyperEnv)

(def ^:dynamic *env*)


;;
(defn intern [val])


;; eval with treshold
;; -> more things become true

;; eval with perspective
;; -> different things become true ig.


;; counterfactuals
;; -> if the perspective would be this, then this would be true

;; In other words, you can ask 'in what world is that true?' and get an answer
;;

;; fuzzy cache, fast and slow system
;;
;; lambdas can be memoized, then it's a fuzzy problem, when your inputs are similar,
;; simply return the cached value.
;; And 'deliberation' is a concept now, saying that if the inputs are relatively dissimilar,
;; then calculate the lambda with new inputs.
;;
;; a hyper lambda encounters arguments, and updates itself based on what it sees.
;; A hyper lambda can skip evaluation and return the first best result,
;; in this case it cleans up the inputs automatically.
;;
;;

;; this is a cleanup operation:
;; cleanup:
;; if the input is sufficiently similar to

(def autoassociate (lambda [x] x))



;; you can ask in what world is this 'not true?', too
;;


;; Seen from this perspective, the technology for coping with large-scale computer systems merges with the technology for building new computer languages, and computer science itself becomes no more (and no less) than the discipline of constructing appropriate descriptive languages.
;; SICP




(defn eval-form-lst [forms]

  )

(defn eval-hyper-lambda [op arguments])


;; eval the args,
;; augment the env make a `frame`
;;
;; Down to primitve procedures and symbols
;;
;; Primitive procedures are clojure functions
;; Symbols are HyperSymbols, which are looked up in the env (memory context)
;;

(defn primitve-op? [op]
  (ifn? op))

(defn eval-sequence [])
(defn extend-environment [])
(defn procedure-paramaters [])
(defn procedure-environment [])

(defn compound-expr? [form]
  (list? form))

(defn apply
  [op arguments]
  (cond
    (primitve-op? op)
    (clj/apply
     op
     (eval-form-lst arguments))
    (lambda? op)
    (eval-hyper-lambda op evaluated-arguments)))

;; conditionals

(defn true? [x])

(defn eval-if [form env]
  (hd/thin
   (apply
    hd/bundle
    (for [b-p (branch-predicates exp env)]
      (if (true? (eval b-p))
        (eval (if-consequent exp) env)
        (eval (if-alternative exp) env))))))

(defn if? [exp]
  )


;; assigment


(defn eval
  [form]
  ;; eval always returns hypervector this can be an
  ;; information mix of multiple outcomes making
  ;; hyperlisp a kind of multiverse process language
  (cond
    (if? form) (eval-if form)
    (compound-expr? form)
    (apply (form->operator form) (form->arguments form))
    (symbol? form) @form
    form))
