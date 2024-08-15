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
            [bennischwerdtner.hd.data :as hdd]
            [bennischwerdtner.hd.codebook-item-memory :as codebook]))


(alter-var-root
 #'hdd/*item-memory*
 (constantly (codebook/codebook-item-memory 1000)))


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
  (sdm/->sdm {:address-count (long 1e6)
              :address-density 0.000003
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
  (hdd/cleanup* (recover m (hd/drop (hdd/clj->vsa* :foo) 0.8) 1))
  ;; (for [n (range 1000)]
  ;;   (let [n (hdd/clj->vsa* n)]
  ;;     (remember m n n)))
  )



;; -----------------------------------------------------
;; effectors

(def actions-item-memory
  (hdd/->TinyItemMemory
   (atom {:left (hdd/clj->vsa :left) :right (hdd/clj->vsa :right)})))

(def cleanup-action #(prot/m-cleanup actions-item-memory %))

(def actions [:left :right])

(def cog-state->action (comp cleanup-action :action-register))

;; ---------------------------------
;; world

(defn update-world [state action]
  (world [state action] state))

;; ---------------------------------------
;; explorer system (not needed, world is easily enumerated)

;;
#_(defn alphabet-finite-state-automaton-play-states
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
       (take (long (* 6 (count alphabet))) actions-xs))))

;; -------------------------------------------

(def m
  (let [m (->memory)]
    (doseq [[state action destination] (map hdd/clj->vsa* (map (fn [[[s a] d]] [s a d]) world))]
      (let [transition (hdd/transition [state action destination])]
        (remember m transition transition)
        (remember m (hdd/clj->vsa* [:* state action]) destination)
        (remember m state state)))
    m))

(let [result-automaton
      (recover m
               (hdd/union
                (hdd/clj->vsa*
                 {{:a #{:right :left}} (hd/permute (hdd/clj->vsa :b))})
                (hdd/clj->vsa* {{:b #{:right :left}}
                                (hd/permute
                                 (hdd/clj->vsa :c))}))
               2)]
  [(prot/m-cleanup actions-item-memory
                   (hdd/automaton-source result-automaton
                                         (hdd/clj->vsa :a)
                                         (hdd/clj->vsa :b)))
   (prot/m-cleanup actions-item-memory
                   (hdd/automaton-source result-automaton
                                         (hdd/clj->vsa :b)
                                         (hdd/clj->vsa
                                          :c)))])
[:right :right]

(defn find-action-structure
  [abc]
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
                 (prot/m-cleanup actions-item-memory
                                 (hdd/automaton-source
                                   result-automaton
                                   (hdd/clj->vsa a)
                                   (hdd/clj->vsa b))))
            abc
            (drop 1 abc))]
    action-structure))

;; this is slow (fixed with gpu codebook impl)
(defn ensure-clj [e]
  (if (hd/hv? e)
    (hdd/cleanup e)
    e))

;; quite trivial since it's in the memory
(defn generate-in-j-domain
  [action-structure j]
  (map ensure-clj
    (reductions
      (fn [action j]
        (let [res
              (recover m (hdd/clj->vsa* [:* j action]) 1)]
          ;; the vocab is in the memory, too so we
          ;; can cleanup from it
          ;; not needed here I guess
          (if-not res
            (ensure-reduced res)
            (recover m res 1))))
      j
      action-structure)))

(defn what-is-the-abc-that-starts-with-j
  [abc j]
  (generate-in-j-domain (find-action-structure abc) j))

(comment
  (hdd/cleanup* (recover m (hdd/clj->vsa* [:* :a :right]) 1))
  (hdd/cleanup* (recover m (hdd/clj->vsa* :a) 1))
  (recover m (hdd/clj->vsa* [:* :c :b]) 2)

  (hdd/cleanup* (recover m (hdd/clj->vsa* [:* :a #{:right :left}]) 1))
  '(:b)

  (hdd/cleanup* (recover m (hdd/clj->vsa* [:* :c #{:right :left}]) 1))
  '(:b)

  (hdd/cleanup* (recover m (hdd/clj->vsa* [:* :c #{:right :left}]) 2))
  '(:b :d)

  ;; if you know the dest
  (let [dest (hdd/clj->vsa* :b)]
    (hdd/cleanup*
     (hdd/intersection
      (recover m (hdd/clj->vsa* [:* :c #{:right :left}]) 2)
      dest)))
  '(:b)

  (hdd/cleanup* (hdd/automaton-source
                 (recover m
                          (hdd/clj->vsa*
                           {{:a #{:right :left}} [:> :b]})
                          2)
                 (hdd/clj->vsa* :a)
                 (hdd/clj->vsa* :b)))

  (let [automaton (recover m
                           (hdd/clj->vsa*
                            {{:a #{:right :left}} [:> :b]})
                           2)]
    (hdd/cleanup* (hdd/clj->vsa* [:*.< automaton :a :_ :b])))

  (find-action-structure [:a :b :c])
  '(:right :right)
  (find-action-structure [:c :b :a])
  '(:left :left)
  (find-action-structure [:c :d :c])
  '(:right :left)

  (hdd/cleanup* (recover m (hdd/clj->vsa* [:* :d :right]) 1))
  '(:e)

  (generate-in-j-domain [:right :right] :j)
  '(:j :k :l)

  (what-is-the-abc-that-starts-with-j [:a :b :c] :j))

;; stuff that works:

(assert
 (= (into [] (what-is-the-abc-that-starts-with-j [:a :b :c] :j))
    [:j :k :l]))

(into [] (what-is-the-abc-that-starts-with-j [:c :b :a] :z))
[:z :y :x]

(into [] (what-is-the-abc-that-starts-with-j [:c :d :c] :j))
[:j :k :j]

(into [] (what-is-the-abc-that-starts-with-j [:c :d :c :d] :j))
[:j :k :j :k]

(into [] (what-is-the-abc-that-starts-with-j [:c :d :c :d :e :f :g] :j))
[:j :k :j :k :l :m :n]

(into [] (what-is-the-abc-that-starts-with-j [:j :k :j] :j))
[:j :k :j]

(into [] (what-is-the-abc-that-starts-with-j [:j :k :j] :o))
[:o :p :o]

(into [] (what-is-the-abc-that-starts-with-j [:j :k] :o))
[:o :p]

;; broke because no action goes between j and j
(into [] (what-is-the-abc-that-starts-with-j [:j :j] :o))
[:o :y]

(time
 (into [] (what-is-the-abc-that-starts-with-j [:j :k :j] :o)))
;;
;; "Elapsed time: 55.769834 msecs"
;; not sure rn where the bottleneck is
;;




;; prevously:
;; 1) ... I only ran this once, not clear if I had luck

;; 2) I ran it a second time and it didn't work. Lol.
;;    I updated so we don't query with the superposition of the alphabet,
;;    We just put more stuff in memory.
;;






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
