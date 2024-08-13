(ns
  tiny-if-abc-got-changed-to-abd-then-what-happenend-to-jkl
  (:require [tech.v3.datatype.functional :as f]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
            [tech.v3.datatype.bitmap :as bitmap]
            [fastmath.random :as fm.rand]
            [fastmath.core :as fm]
            [bennischwerdtner.sdm.sdm :as sdm]
            [bennischwerdtner.hd.binary-sparse-segmented :as
             hd]
            [bennischwerdtner.pyutils :as pyutils]
            [tech.v3.datatype.unary-pred :as unary-pred]
            [tech.v3.datatype.argops :as dtype-argops]
            [bennischwerdtner.hd.data :as hdd]))


;;
;; Followup of what-is-the-abc-that-starts-with-j
;;
;;
;; 👉 it's possible I just had luck. I updated what-is-the-abc-that-starts-with-j after tyring a second time.
;; This is exploratory / example anyway
;;

;;
;; --------------------------------------------------------------------------------------------
;; If abc got changed to abd, what happenend to jkl when it says 'exactly that happenend to me!' ?
;; --------------------------------------------------------------------------------------------
;; (Mitchel and Hofstadter 1988, the copycat project)
;;

;; --------------------------
;;
;; This showcases a tiny region of analogical reason that I can cover using the action sequences of
;; what-is-the-abc-that-starts-with-j
;;
;; --------------------------
;; - The goal is to eventually build up to something that solves copycat domain at least as good as copycat.
;; - Here, I only solve problems of the form
;;   abc->abd ; jkl -> ?
;;
;; in a very limited reach of generality.
;; I don't document this deeply, the code and it's comment are the documentation here.
;;
;;
;;



;; Produce platonic-alphabet:
;; (interfaces not figured out, so reproduce in user space)
;;

;; ----------------------------------------
;; platonic-alphabet is similar to triangle world, but more elements and not circular.
;;
;;

(def alphabet
  (into []
        (map (comp keyword str char)
             (range (int \a) (inc (int \z))))))

(def world
  (into
   ;; not sure if the walls are part of the alphabet,
   ;; but they are part of the world.
   ;; (Need to differentiate identiy and boundaries now).
   {[:a :left] :impossible-wall
    [:z :right] :impossible-wall}
   (concat
    (map (fn [a b] [[a :right] b])
         alphabet
         (drop 1 alphabet))
    (map (fn [a] [[a :identity] a]) alphabet)
    (map (fn [a b] [[a :left] b])
         (reverse alphabet)
         (drop 1 (reverse alphabet))))))

;;
;; a right -> b
;; a left -> 'nothing' (but 'a' because of update-world implementation)

;; -----------------------------------------------
;; memory

(defn ->memory
  []
  (sdm/->sdm {:address-count (long 1e5)
              :address-density 0.00003
              :word-length (long 1e4)}))

(defn remember [sdm addr content]
  (sdm/write sdm addr content 1))

(defn recover-1
  [sdm addr-prime top-k]
  (sdm/lookup sdm addr-prime top-k 1))

(defn recover
  [sdm addr-prime top-k]
  (let [lookup-outcome (recover-1 sdm addr-prime top-k)]
    (when (< 0.1 (:confidence lookup-outcome))
      (some-> lookup-outcome
              :result
              sdm/torch->jvm
              (dtt/->tensor :datatype :int8)))))
;; -----------------------------------------------------
;; effectors
;;
;; - I augment with 'identity' so I capture abc -> abd as
;;   [identity identity right]

(def actions-item-memory
  (hdd/->TinyItemMemory
   (atom {:left (hdd/clj->vsa :left)
          :right (hdd/clj->vsa :right)
          ;; one might consider such a thing
          ;; :identity (hd/->empty)
          ;; then bind with identity results in the input hdv.
          ;; Not sure.
          ;; Perhaps call it :nothing instead of :identity,
          ;; But I thought 'no action' means the world moves on it's own terms.
          ;; 'identity' is a different notion.
          :identity (hdd/clj->vsa :identity)})))

(def cleanup-action #(hdd/m-cleanup actions-item-memory %))

(def actions [:left :right :identity])

(def cog-state->action (comp cleanup-action :action-register))

;; ---------------------------------
;; world

(defn update-world [state action]
  ;; identity is handled correctly, stays at state
  (world [state action] state))

;; ---------------------------------------
;; 1. Create a finite-state automaton, update an sdm with transition triplets.
;;
;; (it's just the world)

(def m
  (let [m (->memory)]
    (doseq [transition
              (map (comp hdd/transition hdd/clj->vsa*)
                (map (fn [[[s a] d]] [s a d]) world))]
      (remember m transition transition))
    m))


;;
;; -----------------------------------------------------------------------
;; Idea 2:
;;
;; source domain:
;; ----------------
;;
;; source:              [ a b c ]
;; destination:         [ a b d ]
;;
;; action-structure:    [ identity identity right ]
;;
;;
;; target domain:
;; -----------------
;;
;;
;; source:              [ j k l ]
;; destination:         [ & _]
;;
;; action-structure:    [ identity identity right ]
;;
;;
;; generate, using action-structure:
;;
;; destination:         [ j k m ]
;;
;;
;;
;;
;; - This I can do with the material at hand, the only thing I need is an identity action
;; -----------------------------------------------------------------------


;;
;; Note the query:
;;
;;
;; [ source ⊙ (set of all possible transitions) ] -> destination
;;
;;
;; This is akin to a generlized state in an non-deterministic finite-state automaton.
;; Here, source and destination are given, and the action is generalized (all possible).
;;
;;
;;  When querying for the transition [a b],
;;
;;
;; What is in memory:
;;
;;
;;           a  <-  b    ->     c
;;            left      right
;;                       ->
;;                      identity b
;;
;; What we query with:
;;
;;
;;           c  <-  b    ->     c
;;            left      right
;;                       ->
;;                      identity c
;;
;; The only sensical output is.
;;
;;
;;            b   ->    c
;;               right
;;
;;
;; Perhaps it is fair to say we 'collapse' the superposition queries into the 'possible' ones (the ones in memory).
;;

;;
;;  copy stuff from what_is_the_abc_that_starts_with_j.clj
(defn find-action-structure
  [m abc]
  (let [result-automaton (recover
                          m
                          ;; query for the full
                          ;; automaton
                          (apply hdd/union
                                 (map (fn [a b]
                                        (hdd/transition
                                         (hdd/clj->vsa a)
                                         ;; query with all
                                         ;; actions. This is
                                         ;; ~ equivalent to
                                         ;; running all
                                         ;; finite state
                                         ;; automatons and
                                         ;; see which one
                                         ;; led to the
                                         ;; outcome. But
                                         ;; here, we use the
                                         ;; memory to say
                                         ;; which
                                         ;; transitions
                                         ;; exist. You see
                                         ;; here, it is
                                         ;; advantagous to
                                         ;; keep the count
                                         ;; of discrete
                                         ;; actions low.
                                         (hdd/clj->vsa*
                                          (into #{} actions))
                                         ;;
                                         (hdd/clj->vsa b)))
                                      abc
                                      (drop 1 abc)))
                          ;; +1 from trial and error
                          (+ 2 (count abc)))
        action-structure
        ;; use the automaton to find the transitions
        ;; between input seq
        (map (fn [a b]
               (hdd/m-cleanup actions-item-memory
                              (hdd/automaton-source
                               result-automaton
                               (hdd/clj->vsa a)
                               (hdd/clj->vsa b))))
             abc
             (drop 1 abc))]
    action-structure))

(defn memory-automaton-destination
  [m source action]
  ;; I didn't create a [⊙ source action] -> destination
  ;; memory, but it would be useful here.
  ;;
  (some->
    (recover
      m
      ;; querying with superposition
      (hd/bind*
        [(hdd/clj->vsa* source) (hdd/clj->vsa* action)
         ;; That's so swag c'mon (saying whole alphabet is possible as target)
         (hd/permute (hdd/clj->vsa* (into #{} alphabet)))])
      ;;
      ;; was 2, made it 3. +1 for each action in the
      ;; training set? Not sure.
      3)
    (hdd/automaton-destination (hdd/clj->vsa* source)
                               (hdd/clj->vsa* action))))



(comment
  ;; Step 1: Find the 'translational action structure'
  ;;
  (let [source [:a :b :c]
        destination [:a :b :d]
        source-target-action-structure
        (map (comp #(find-action-structure m %) vector)
             source
             destination)]
    source-target-action-structure)
  '((:identity) (:identity) (:right))
  ;; Step 2: Gen with target 'input'
  ;;
  (let [source-target-action-structure
        '((:identity) (:identity) (:right))
        target-input [:j :k :l]]
    (hdd/cleanup (memory-automaton-destination
                  m
                  (hdd/clj->vsa* :a)
                  (hdd/clj->vsa* :identity)))
    (hdd/cleanup (memory-automaton-destination
                  m
                  (hdd/clj->vsa* :j)
                  (hdd/clj->vsa* :identity)))
    (map (fn [source [action]]
           (hdd/cleanup (memory-automaton-destination
                         m
                         (hdd/clj->vsa* source)
                         (hdd/clj->vsa* action))))
         target-input
         source-target-action-structure))
  '(:j :k :m))

(defn
  tiny-if-abc-got-changed-to-abd-then-what-happenend-to-jkl
  [abc abd jkl]
  (let [source abc
        destination abd
        source-target-action-structure
          (map (comp #(find-action-structure m %) vector)
            source
            destination)
        target-input jkl]
    (map (fn [source [action]]
           (hdd/cleanup (memory-automaton-destination
                          m
                          (hdd/clj->vsa* source)
                          (hdd/clj->vsa* action))))
      target-input
      source-target-action-structure)))


;; -----------------------------

(tiny-if-abc-got-changed-to-abd-then-what-happenend-to-jkl
 [:a :b :c]
 [:a :b :d]
 [:j :k :l])
'(:j :k :m)

(tiny-if-abc-got-changed-to-abd-then-what-happenend-to-jkl
 [:b :b :b]
 [:a :b :c]
 [:j :j :j])
'(:i :j :k)



(tiny-if-abc-got-changed-to-abd-then-what-happenend-to-jkl
 [:x :w :x]
 [:w :x :w]
 [:j :i :j])
'(:i :j :i)

(tiny-if-abc-got-changed-to-abd-then-what-happenend-to-jkl
 [:a :b :a]
 [:b :a :b]
 [:j :i :j])
'(:k :h :k)

(tiny-if-abc-got-changed-to-abd-then-what-happenend-to-jkl
 [:a :b :a]
 [:b :a :b]
 [:i :j :i])
'(:j :i :j)

;; Fails for relationships other than 1 away of course:
(tiny-if-abc-got-changed-to-abd-then-what-happenend-to-jkl
 [:a :x :a]
 [:a :a :a]
 [:b :x :b])
'(:b nil :b)

;; .. although this 'partial overlap' structure is joyful by itself

(tiny-if-abc-got-changed-to-abd-then-what-happenend-to-jkl
 [:u :v :v]
 [:u :u :u]
 [:u :v :v])
'(:u :u :u)

(tiny-if-abc-got-changed-to-abd-then-what-happenend-to-jkl
 [:u :v :v]
 [:t :u :u]
 [:a :a :a])
'(nil nil nil)
;; because there is nothing in front of :a

(tiny-if-abc-got-changed-to-abd-then-what-happenend-to-jkl
 [:u :v :v]
 [:t :u :u]
 [:b :b :b])
'(:a :a :a)

(tiny-if-abc-got-changed-to-abd-then-what-happenend-to-jkl
 [:u :v :v]
 [:t :u :u]
 [:c :b :b])
'(:b :a :a)


;; single minded in a way:

(tiny-if-abc-got-changed-to-abd-then-what-happenend-to-jkl
 [:x :b :x :b :b]
 [:y :c :w :a :a]
 [:b :b :b :b :b])
'(:c :c :a :a :a)



;; --------------------------------------------------
;; Discussion:
;;
;; While this might not be an element of a final hyper copycat, this is a start
;; at showcasing at least the general direction of things.
;;
;; - the analogy engine is enterily lifted up by hdc/vsa
;; - + an SDM
;; - this means that the whole approach has the potential to express ambiguity and so forth,
;;   this is levaraged here by finding the transition between to symbols in superposition.
;; - It would be cool to find elegant ways of absracting such things,
;;   such that we program with multiple possible, 'possible' 'known worlds', not only making analogies,
;;   but mixing and matching analogies.
;; - It arguably solves similar problems to logic programing or probabilistic programing,
;;   but here, ~ everything is done with vector + / - / * and permuatations.
;; - The speed with which we went from 'neuronal model' to (tiny) 'subcognitive concept space programming' is fun.
;;










(comment
  (find-action-structure m [:a :a])
  (hdd/cleanup
   (let [result-automaton
         (recover m
                  (hdd/union
                   (hdd/transition (hdd/clj->vsa :a)
                                   (hdd/clj->vsa
                                    :identity)
                                   (hdd/clj->vsa :a))
                   (hdd/transition (hdd/clj->vsa :a)
                                   (hdd/clj->vsa :left)
                                   (hdd/clj->vsa :a))
                   (hdd/transition (hdd/clj->vsa :a)
                                   (hdd/clj->vsa :right)
                                   (hdd/clj->vsa :a)))
                  2)]
     (hdd/automaton-source result-automaton
                           (hdd/clj->vsa :a)
                           (hdd/clj->vsa :a))))
  (hdd/cleanup (let [result-automaton
                     (recover
                      m
                      (apply hdd/union
                             (map (fn [a b]
                                    (hdd/transition
                                     (hdd/clj->vsa a)
                                     (hdd/clj->vsa*
                                      (into #{} actions))
                                     (hdd/clj->vsa b)))
                                  [:a]
                                  [:a]))
                      2)]
                 (hdd/automaton-source result-automaton
                                       (hdd/clj->vsa :a)
                                       (hdd/clj->vsa :a))))
  (hdd/cleanup (let [result-automaton
                     (recover
                      m
                      (apply hdd/union
                             (map (fn [a b]
                                    (hdd/transition
                                     (hdd/clj->vsa a)
                                     (hdd/clj->vsa*
                                      (into #{} actions))
                                     (hdd/clj->vsa b)))
                                  [:b]
                                  [:b]))
                      3)]
                 (hdd/automaton-source result-automaton
                                       (hdd/clj->vsa :b)
                                       (hdd/clj->vsa :b)))))




(comment
  (let [source [:b]
        destination [:a]
        source-target-action-structure
        (map (comp #(find-action-structure m %) vector)
             source
             destination)]
    source-target-action-structure)
  (hdd/cleanup (let [result-automaton
                     (recover
                      m
                      (apply hdd/union
                             (map (fn [a b]
                                    (hdd/transition
                                     (hdd/clj->vsa a)
                                     (hdd/clj->vsa*
                                      (into #{} actions))
                                     (hdd/clj->vsa b)))
                                  [:b]
                                  [:a]))
                      4)]
                 (hdd/automaton-source result-automaton
                                       (hdd/clj->vsa :b)
                                       (hdd/clj->vsa :a))))
  (hdd/cleanup (let [result-automaton
                     (recover m
                              (hdd/transition
                               (hdd/clj->vsa :b)
                               (hdd/clj->vsa :left)
                               (hdd/clj->vsa :a))
                              2)]
                 (hdd/automaton-source result-automaton
                                       (hdd/clj->vsa :b)
                                       (hdd/clj->vsa :a)))))




;;
;;
;;
;; - Extract action structures ?
;;
;; -----------------------------------------------------------------------
;; Idea 1:
;;
;;
;; In source domain:
;;
;;
;; source:       [ a b c]
;;               [ right right ],  :a domain
;;
;; destination:  [ a b d ]
;;               [ right _ ]
;;               [ right (:> right what-happenend-to-c) ]
;;
;; what-happenend-to-c:
;;
;; source:                 c
;; destinations:           d
;; action-structure:       :right


;;
;; where :> is a chain of actions
;;


;; In target domain:

;; source:         [ j k l ]
;;                 [ right right], :j domain
;;
;; destination:    [ & _ ]
;;
;; Gen with        [ right (:> right what-happenend-to-c) ]
;;                 [ right (:> right right) ]
;;
;;
