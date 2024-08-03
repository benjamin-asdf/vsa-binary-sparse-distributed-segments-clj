(ns substring-automaton
  (:require [tech.v3.datatype.functional :as f]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
            [tech.v3.datatype.bitmap :as bitmap]
            [fastmath.random :as fm.rand]
            [fastmath.core :as fm]
            [bennischwerdtner.hd.binary-sparse-segmented :as
             hd]
            [tech.v3.datatype.unary-pred :as unary-pred]
            [tech.v3.datatype.argops :as dtype-argops]
            [bennischwerdtner.hd.data :as hdd]))

;; I didn't make this work yet


;; Using a nonderministic finte-state automaton for substring search
;; --------------------------------------------------------------------
;;

;; - each position of a symbol in the base string is modeled as a unique state of the automaton S
;; - hello generates an automaton with 6 states
;; - transitions are the base string symbols {b1,b2,...,bn}

(def base-string "hello")
;; (def )

(def automaton
  ;; has unique state for each position
  ;; then position-1 ⊙ symbol -> position-2
  ;;
  ;; So you can see that you
  ;;
  (hdd/finite-state-automaton-1
    (for [[n symbol] (map-indexed vector base-string)]
      (hdd/clj->vsa* [n symbol (inc n)]))))

(hdd/cleanup*
 (hdd/automaton-destination
  automaton
  (hdd/clj->vsa* 0)
  (hdd/clj->vsa* \h)))

'(1)

(def q "hel")


(hdd/cleanup*
 (hdd/automaton-destination
  automaton
  (hdd/clj->vsa* (into #{} (range (count base-string))))
  (hdd/clj->vsa* \e)))
'(2)

(hdd/cleanup*
 (let [p
       (hdd/automaton-destination
        automaton
        (hdd/clj->vsa*
         (into #{} (range (count base-string))))
        (hdd/clj->vsa* \e))
       p (some->
          (hdd/cleanup p)
          (hdd/clj->vsa))
       ]
   (hdd/automaton-destination
    automaton
    p
    (hdd/clj->vsa* \l))))

(hdd/automaton-destination
 (hdd/finite-state-automaton-1
  [(hdd/clj->vsa* [0 \a 1])])
 (hdd/clj->vsa* 0)
 (hdd/clj->vsa* \a))


(let
    [base-string (map char (range (int \a) (inc (int \z))))
     automaton (hdd/finite-state-automaton-1
                (for [[n symbol]
                      (map-indexed vector base-string)]
                  (hdd/clj->vsa* [n symbol (inc n)])))]
  (hdd/cleanup-verbose
   (hdd/automaton-destination
    automaton
    (hdd/clj->vsa* (into #{} (range 10)))
    (hdd/clj->vsa* \a)))
    ;; (hdd/cleanup*
    ;;  (hdd/automaton-destination
    ;;   (hdd/finite-state-automaton-1
    ;;    [(hdd/clj->vsa* [0 \a 1])])
    ;;   ;; automaton
    ;;   (hdd/clj->vsa* (into #{} (range 5)))
    ;;   (hdd/clj->vsa* \a)))
  )


(let [x (hd/unbind
          (hdd/intersection
            [(hdd/clj->vsa* (into
                              {}
                              (map-indexed vector)
                              "abc"
                              ;; (map char (range
                              ;; (int \a)
                              ;;                  (inc
                              ;;                  (int
                              ;;                  \z))))
                            ))
             (hdd/clj->vsa*
               (into {} (map-indexed vector) "abc"))])
          (hdd/clj->vsa* (into #{} "abc")))]
  (map (fn [i] (hd/similarity x (hdd/clj->vsa* i)))
       (range 5)))

'(0.5 0.45 0.05 0.0 0.0)






;; (0 1)






















1.0

(f/sum (hdd/clj->vsa* 0))

(hd/similarity (hdd/clj->vsa* 0) (hdd/clj->vsa* 0))










(into #{} (range 5))

(=
 (hdd/clj->vsa* (into #{} (range 5)))
 (hdd/clj->vsa* #{0 1 2}))



(let
    [base-string "abcde"
     ;; (map char (range (int \a) (inc (int \z))))
     automaton
     ;; state0 ⊙ a -> state1
     (hdd/finite-state-automaton-1
      (for [[n symbol] (map-indexed vector base-string)]
        (hdd/clj->vsa* [n symbol (inc n)])))
     query-string "abc"]
    (reduce
     (fn [state query]
       (cond (not query) (ensure-reduced state)
             (not state) (ensure-reduced false)
             :else (some-> (hdd/automaton-destination
                            automaton
                            state
                            query)
                           ;; cleaning up at each
                           ;; step like they also
                           ;; do in the paper such
                           ;; a thing can be done
                           ;; with a resonator
                           ;; network
                           hdd/cleanup
                           hdd/clj->vsa)))
     ;;
     ;; in the paper,
     ;; the initial state is the superposition of all
     ;; states for the base-string
     ;;
     ;; but here, the signal noise ration isn't capable of handling this,
     ;; so I first search through chunks of the atomaton for a start state
     ;;
     ;;
     ;;
     #_(hdd/clj->vsa* (into #{} (range (count base-string))))

     ;; wip
     (recur [state
             query (first (hdd/clj->vsa* (into [] query-string)))]
            )
     (hdd/clj->vsa* (into #{} (range (count base-string))))
     (some-> (hdd/automaton-destination
              automaton
              state
              query)
             ;; cleaning up at each
             ;; step like they also
             ;; do in the paper such
             ;; a thing can be done
             ;; with a resonator
             ;; network
             hdd/cleanup
             hdd/clj->vsa)





     ;; hypersymbols for the query string
     (hdd/clj->vsa* (into [] query-string))))














(hdd/cleanup*
 (let [base-string "abc"
       ;; (map char (range (int \a) (inc (int \z))))
       automaton
       ;; state0 ⊙ a -> state1
       (hdd/finite-state-automaton-1
        (for [[n symbol] (map-indexed vector
                                      base-string)]
          (hdd/clj->vsa* [n symbol (inc n)])))
       query-string "a"]
   (reduce (fn [state query]
             (hdd/automaton-destination automaton
                                        state
                                        query))
           ;; initial generalized state
           (hdd/clj->vsa* (into #{} (range (count base-string))))
           ;; hypersymbols for the query string
           (hdd/clj->vsa* (into [] query-string)))))
