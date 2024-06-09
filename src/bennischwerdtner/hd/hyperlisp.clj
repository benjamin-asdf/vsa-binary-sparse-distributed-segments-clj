(ns bennischwerdtner.hd.hyperlisp
  (:refer-clojure :exclude [apply eval symbol? intern])
  (:require
   [bennischwerdtner.hd.binary-sparse-segmented :as
    hd]

   ))


;; --------------------------------
;; Lambda Calculus With High Dimensional Computing
;; --------------------------------
;;
;; Lisp
;; Recursive Functions of Symbolic Expressions and Their Computation by Machine, Part I
;; http://www-formal.stanford.edu/jmc/recursive.html

;;
;;

;; resolves to more than 1 thing
(defprotocol IHyperSymbol)






;; Seen from this perspective, the technology for coping with large-scale computer systems merges with the technology for building new computer languages, and computer science itself becomes no more (and no less) than the discipline of constructing appropriate descriptive languages.
;; SICP


(defn eval [form env])


;; eval the args,
;; augment the env make a `frame`
;;
;; Down to primitve procedures and symbols
;;
;; Primitive procedures are clojure functions
;; Symbols are HyperSymbols, which are looked up in the env (memory context)
;;


(defn apply [op arguments]

  )
