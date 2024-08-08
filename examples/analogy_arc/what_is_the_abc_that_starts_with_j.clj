(ns what-is-the-abc-that-starts-with-j
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

;; -----------------------------------
;; Series:
;; 1) triangle_world.clj.
;; 2) k_fold_triangle.clj
;; 3) platonic-alphabet-v1
;; 4) what_is_the_abc_that_starts_with_j.clj
;; 5) tiny_if_abc_got_changed_to_abd_then_what_happenend_to_jkl.clj
;;
;;
;;
;; this is with segment count = 20 btw
;; with 100 there are other tradeoffs with memory, similarity etc.



;; The plan:
;; ----------------------------------------------------
;;
;;
;; - Given 'abc', extract [:right :right :right] ('action trajectory'?)
;; - Given platonic-alphabet, an action trajectory, and a start character, generate a string
;;   (a world trajectory)
;;


;; Produce platonic-alphabet:
;; (interfaces not figured out, so reproduce in user space)
;;

;; ----------------------------------------
;; platonic-alphabet is similar to triangle world, but more elements and not circular.
;;
;;

(def alphabet (into [] (map (comp keyword str char) (range (int \a) (inc (int \z))))))

(def world
  (into {}
        (concat (map (fn [a b] [[a :right] b])
                  alphabet
                  (drop 1 alphabet))
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

(comment
  (def m (->memory))
  (remember m (hdd/clj->vsa* :foo) (hdd/clj->vsa* :foo))
  (hdd/cleanup* (recover m (hd/drop (hdd/clj->vsa* :foo) 0.8) 1)))



;; -----------------------------------------------------
;; effectors

(def actions-item-memory
  (hdd/->TinyItemMemory
   (atom {:left (hdd/clj->vsa :left) :right (hdd/clj->vsa :right)})))

(def cleanup-action #(hdd/m-cleanup actions-item-memory %))

(def actions [:left :right])

(def cog-state->action (comp cleanup-action :action-register))

;; ---------------------------------
;; world

(defn update-world [state action]
  (world [state action] state))

;; ---------------------------------------
;; explorer system
;;
;; I can scrap the explorer for the simple approach:
;;
;; I. Create a finite-state automaton, update an sdm with transition triplets.
;;

(defn alphabet-finite-state-automaton-play-states
  [alphabet]
  (let [actions-xs
          (cycle (concat
                   (repeat (+ 2 (count alphabet)) :right)
                   (repeat (+ 2 (count alphabet)) :left)))]
    (reductions
      (fn [{:keys [world]} action]
        (let [next-world (update-world world action)]
          {:transition [world action next-world]
           :world next-world}))
      {:world :a}
      (take (long (* 2.5 (count alphabet))) actions-xs))))

;; -------------------------------------------

(def m
  (let [m (->memory)]
    (doseq
      [transition
         (map (comp hdd/transition hdd/clj->vsa*)
           (keep
             :transition
             (alphabet-finite-state-automaton-play-states
              alphabet)))]
      (remember m transition transition))
    m))

(let [result-automaton
        (recover m
                 (hdd/union
                   (hdd/clj->vsa* {{:a #{:right :left}}
                                     (hd/permute
                                       (hdd/clj->vsa :b))})
                   (hdd/clj->vsa* {{:b #{:right :left}}
                                     (hd/permute
                                       (hdd/clj->vsa :c))}))
                 2)]
  [(hdd/m-cleanup actions-item-memory
                  (hdd/automaton-source result-automaton
                                        (hdd/clj->vsa :a)
                                        (hdd/clj->vsa :b)))
   (hdd/m-cleanup actions-item-memory
                  (hdd/automaton-source result-automaton
                                        (hdd/clj->vsa :b)
                                        (hdd/clj->vsa
                                          :c)))])
[:right :right]


;; ...
;; recovering a non deterministic finite state automaton from the sdm by querying in superposition
;; and retrieving multiple values per key
;;


(defn what-is-the-abc-that-starts-with-j
  [abc j]
  (let [result-automaton
          (recover m
                   ;; query for the full automaton
                   (apply hdd/union
                     (map (fn [a b]
                            (hdd/clj->vsa*
                              {{a #{:right :left}}
                                 (hd/permute (hdd/clj->vsa
                                               b))}))
                       abc
                       (drop 1 abc)))
                   (count abc))
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
    ;; use action seq and start symbol to generate from
    ;; the automaton triplets in the memory.
    ;; (at this point a sequence memory would be handy)
    ;;
    ;; (this problem is rather trivial given the
    ;; previous installments of this series)
    (reductions
      (fn [j action]
        (some->
          (recover m
                   ;; querying with superposition
                   (hd/bind* [(hdd/clj->vsa* j)
                              (hdd/clj->vsa* action)
                              ;; That's so swag c'mon
                              (hd/permute
                               (hdd/clj->vsa*
                                (into #{} alphabet)))])
                   ;; I came to 2 by trying out
                   2)
          (hdd/automaton-destination (hdd/clj->vsa* j)
                                     (hdd/clj->vsa* action))
          ;; you might need to cleanup in between here
          ;; could be handled by an SDM, too
          hdd/cleanup
          hdd/clj->vsa))
      (hdd/clj->vsa j)
      action-structure)))

;; stuff that works:

(into [] (map hdd/cleanup (what-is-the-abc-that-starts-with-j [:a :b :c] :j)))
[:j :k :l]

(into [] (map hdd/cleanup (what-is-the-abc-that-starts-with-j [:c :b :a] :z)))
[:z :y :x]

(into [] (map hdd/cleanup (what-is-the-abc-that-starts-with-j [:c :d :c] :j)))
[:j :k :j]

(into [] (map hdd/cleanup (what-is-the-abc-that-starts-with-j [:c :d :c :d] :j)))
[:j :k :j :k]

(into [] (map hdd/cleanup (what-is-the-abc-that-starts-with-j [:c :d :c :d :e :f :g] :j)))
[:j :k :j :k :l :m :n]

(into [] (map hdd/cleanup (what-is-the-abc-that-starts-with-j [:j :k :j] :j)))
[:j :k :j]

(into [] (map hdd/cleanup (what-is-the-abc-that-starts-with-j [:j :k :j] :o)))
[:o :p :o]

(into [] (map hdd/cleanup (what-is-the-abc-that-starts-with-j [:j :k] :o)))
[:o :p]

;; error, doesn't recover a transition from j to j
 (into [] (map hdd/cleanup (what-is-the-abc-that-starts-with-j [:j :j] :o)))
;; error




;; ... I only ran this once, not clear if I had luck



;; -------------------------------


;;
;; - The concept of 'action structure' as a way to say what the 'structure of an analogy is', is in my opinion coherent.
;; - it's whatever we can extract from [:a :b :c], then use it to generate [:j :k :l] given :j, [:x :y :z] given :x and so forth.
;;
;;
;; - makes use of non-deterministic finite state automata and an SDM that supports multiple values per key.
;;
;; - Admittedly, I suppose such a thing could be expressed with symbols, a database and a query language like datalog
;;
;; - A richer action vocabulary would in principle scale this
;;
;; - Presumably, at some point you will need a process that selects (frames) depending on the problem,
;;   which kinds of actions are applicable
;;  (that is tackled in 'copycat project' in interesting ways)
;;
;;
;; - I wonder how you bring such a thing together with making analogies like triangle world
;; (eventually scaling to many 'known worlds')
;;
;; - Can I extract and action structure and sort of say 'this fits triangle world'?
;;
;; - [:a :b :c :a] would fit triangle world
;; - [:a :b :c :d] would not fit triangle world
;;
;; - wouldn't it be sick to figure out something that can do that in superposition, too?
;;  (check multiple worlds)
;;  (maybe the overlap/intersection of stuff will help)
;;
;;





































(comment
  (hdd/cleanup* (recover m
                         (hdd/clj->vsa* {#{:j :p} #{:right
                                                    :left}})
                         4))
  (hdd/cleanup* (hd/unbind
                 (recover m
                          (hdd/clj->vsa* {{:j :right}
                                          (hd/permute
                                           (hdd/clj->vsa
                                            :k))})
                          1)
                 (hd/bind (hd/permute (hdd/clj->vsa :k))
                          (hdd/clj->vsa :j))))
  (hdd/cleanup* (hd/unbind
                 (recover m
                          (hdd/clj->vsa*
                           {{:a #{:right :left}}
                            (hd/permute (hdd/clj->vsa
                                         :b))})
                          2)
                 (hd/bind (hd/permute (hdd/clj->vsa :b))
                          (hdd/clj->vsa :a))))
  (hdd/cleanup* (hd/unbind
                 (recover m
                          (hdd/clj->vsa*
                           {{#{:a :b} #{:right :left}}
                            (hd/permute (hdd/clj->vsa
                                         #{:b :c}))})
                          2)
                 (hd/bind (hd/permute (hdd/clj->vsa :b))
                          (hdd/clj->vsa :a))))
  (let [result-automaton
        (recover m
                 (hdd/union
                  (hdd/clj->vsa*
                   {{:a #{:right :left}}
                    (hd/permute (hdd/clj->vsa :b))})
                  (hdd/clj->vsa*
                   {{:b #{:right :left}}
                    (hd/permute (hdd/clj->vsa :c))}))
                 4)]
    (hdd/cleanup*
     (hd/unbind (recover
                 m
                 (hdd/union
                  (hdd/clj->vsa*
                   {{:a #{:right :left}}
                    (hd/permute (hdd/clj->vsa :b))})
                  (hdd/clj->vsa*
                   {{:b #{:right :left}}
                    (hd/permute (hdd/clj->vsa :c))}))
                 4)
                (hd/bind (hd/permute (hdd/clj->vsa :c))
                         (hdd/clj->vsa :b)))))
  (let [result-automaton
        (recover m
                 (hdd/clj->vsa* {{:n #{:right :left}}
                                 (hd/permute
                                  (hdd/clj->vsa :m))})
                 1)]
    (hdd/cleanup* (hdd/automaton-source result-automaton
                                        (hdd/clj->vsa :n)
                                        (hdd/clj->vsa :m))))
  '(:left))
