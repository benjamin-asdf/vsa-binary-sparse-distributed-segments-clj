
;; this doesn't make that much sense:
(comment
  (doseq [n (map hdd/clj->vsa* (range 7))]
    (sdm/write sdm n n 1))

  ;; this turns out to work with
  ;; segment-count 100
  ;; and dropping 0.8 in the combination

  (hdd/cleanup*
   (:result (sdm/lookup sdm (reduce combination (range 7)) 7 1)))
  '(0 1 2 3 5 6)

  (map #(hdd/clj->vsa* [:?= (reduce combination (range 7)) %]) (range 7))
  (0.06 0.04 0.08 0.09 0.02 0.05 0.08)

  (hd/similarity
   (hd/bind* (map hdd/clj->vsa* (range 7)))
   (reduce combination (range 7)))



  (hd/similarity
   (hdd/clj->vsa* [:* :a :b])
   (f/sum (combination :a :b)))




  )









(defn ->decider
  [symbolic-codebook]
  (let [item-memory
        (hdd/->TinyItemMemory
         (atom
          (into {}
                (for [k symbolic-codebook]
                  [k (hdd/clj->vsa* k)]))))]
    (reify
      Decider
      (decide [this input]
        (prot/m-cleanup item-memory input)))))

(def action-decider (->decider [:left :right :halt :-]))
(def state-decider (->decider [:s1 :s0]))
(def output-symbol-decider (->decider [0 1 :b false true]))


(comment
  (def x-reportior (hdd/clj->vsa* #{:left :right}))
  (def y-reportior (hdd/clj->vsa* #{[:> :left 1] [:> :right 1]}))
  (def z-reportior (hdd/clj->vsa* #{[:> :left 2] [:> :right 2]}))

  (def s (hdd/clj->vsa* [:*> :left :right :left]))

  (hd/bind*
   [(hdd/clj->vsa* [:*> :left :right :left])
    (hdd/clj->vsa* :a)])

  (f/sum
   (hdd/clj->vsa* [:*> :left :right :left]))

  (f/sum
   (hd/bind*
    [(hdd/clj->vsa* [:*> :left :right :left])
     (hdd/clj->vsa* :a)]))

  ;;
  ;; the 'problem' here is that the bind essentially thins,
  ;; dropping down the contribution of everything in the sumset
  ;;
  (f/sum (hd/bind* [z-reportior y-reportior]))
  100.0

  ;; the resultant hdv, now has 1/4th of the info for [:left 1], etc.


  (hd/unbind s (hd/bind* [z-reportior y-reportior]))
  ;; ~ x-reportior

  (hd/similarity
   x-reportior
   (hd/unbind s (hd/bind* [z-reportior y-reportior])))
  0.26
  ;; I guess because of some symmetry, we come to 1/4th here, too
  ;; this overlap comes from the 1/4th [:left] contribution
  ;;



  )





;; ---------------------------------
;; Showing that this really has to do with 'liquid':
;;
;; this actually ends up unbing with 'nothing', which is identiy
;;


(def tofifee-domain
  (hdd/clj->vsa* {:ground :caramel :surface :chocolate}))

(hdd/cleanup*
 (hd/unbind
  ;; ~ nothing
  (hd/unbind rocks-domain
             ;; intersection
             ;; ~ nothing !
             (hdd/intersection-1 [bread-domain tofifee-domain]))
  (hdd/clj->vsa* :ground)))

(f/sum (hdd/intersection-1 [bread-domain tofifee-domain]))

(= (hdd/clj->vsa* :foo) (hd/unbind (hdd/clj->vsa* :foo) (hd/->empty)))


;; -------------------------------------------------------------


;; to show that you can juggle these things a little,
;; let's query the union:

(hdd/cleanup*
 (hd/unbind
  ;; ~ {:ground (⊕ :rocks :bread)}
  (hd/unbind (hdd/union rocks-domain bread-domain)
             ;; intersection
             ;; ~ {:surface liquid}
             (hdd/intersection-1 [bread-domain
                                rocks-domain]))
  (hdd/clj->vsa* :ground)))
'(:bread :rocks)


;; the difference is also meaningful
(hdd/cleanup*
 (hd/unbind
  (hd/unbind (hdd/difference rocks-domain bread-domain)
             ;; intersection
             ;; ~ {:surface liquid}
             (hdd/intersectiuon [bread-domain
                                rocks-domain]))
  (hdd/clj->vsa* :ground)))
'(:rocks)

;; other way around:
(hdd/cleanup*
 (hd/unbind
  (hd/unbind (hdd/difference bread-domain rocks-domain)
             ;; intersection
             ;; ~ {:surface liquid}
             (hdd/intersection-1 [bread-domain
                                rocks-domain]))
  (hdd/clj->vsa* :ground)))
'(:bread)


;; --------------------------------------------------------------------

(comment
  (->
   (recover
    item-memory
    (hdd/automaton-destination
     bread-domain
     (hdd/clj->vsa* {:bread {:surface :empty}})
     (hdd/clj->vsa* {:spread butter})))
   ;; asking what bread do I have after
   (hd/unbind
    (hdd/clj->vsa* :bread))
   (hd/unbind
    (hdd/clj->vsa* :surface))
   (hdd/cleanup*))
  '(:butter :liquid))

;; -----------------------------------------
;; Lets' say that the system has associated lava with liquid
;;

(def lava (hd/thin (hd/superposition liquid (hdd/clj->vsa :lava))))

;; hm, I can recover bread of course when I query with the supperposition of lava and butter
(hdd/cleanup*
 (hd/unbind (recover
             item-memory
             (hdd/automaton-source
              bread-domain
              (hdd/clj->vsa* {:bread {:surface
                                      #{lava butter}}})
              (hdd/clj->vsa* {:spread #{lava butter}})))
            (hdd/clj->vsa* {:surface :empty})))
'(:bread)

;; that's not remarkable
;;

(def bread-domain
  (hdd/finite-state-automaton-1
    (remember-leaves!
      (hdd/clj->vsa*
        [[#{:bread {:surface :empty}} {:spread butter}
          #{:bread {:surface butter}}]
         [#{:bread {:surface butter}} {:spread butter}
          #{:bread {:surface butter :thick? true}}]
         [#{:bread {:surface butter}} {:scrub-off butter}
          #{:bread {:surface butter :thin? true}}]]))))

(def rocks-domain
  (hdd/finite-state-automaton-1
    (remember-leaves!
      (hdd/clj->vsa*
       [[#{:rocks {:surface :empty}}
          {:spread lava}
          #{:rocks {:surface lava}}]
         [#{:rocks {:surface lava}} {:spread lava}
          #{:rocks {:surface lava :thick? true}}]
         [#{:rocks {:surface lava}}
          {:scrub-off lava}
          #{:rocks {:surface lava :thin? true}}]]))))

(hdd/cleanup*
 (recover item-memory
          (hdd/automaton-source
           bread-domain
           ;; note the lack of bread
           (hdd/clj->vsa* {:surface lava})
           (hdd/clj->vsa* {:spread lava}))))
'(:rocks :bread)


;; -------------------------
;; This works because {:spread lava} and {:spread butter} are similar
;;
;; This isn't much more remarkable that 'what is the dollar of mexico'
;; but uses a finite state automaton and ambiguity.
;;

;; If I query the intersection, I get rocks out:
(hdd/cleanup*
 (recover item-memory
          (hdd/automaton-source
           (hdd/intersection bread-domain rocks-domain)
           ;; note the lack of bread
           (hdd/clj->vsa* {:surface lava})
           (hdd/clj->vsa* {:spread lava}))))
'(:rocks)

;; querying the intersection for butter yields bread
(hdd/cleanup*
 (recover item-memory
          (hdd/automaton-source
           (hdd/intersection-1 [bread-domain
                              rocks-domain])
           ;; note the lack of bread
           (hdd/clj->vsa* {:surface butter})
           (hdd/clj->vsa* {:spread butter}))))
'(:bread)


;;
;; set difference amplifies the difference betweent the domains
;;

;; these are non-deterministic outcomes, coinflips I guess
;; It depends on the seed vectors going in, so once the seedvectors are set, this is deterministic
;; (it also depends on what is in the item memory)
(hdd/cleanup* (recover
               item-memory
               (hdd/automaton-source
                (hdd/difference rocks-domain bread-domain)
                (hdd/clj->vsa* {:thin? true})
                (hdd/clj->vsa* {:scrub-off lava}))))
'(:rocks)

(hdd/cleanup* (recover
                item-memory
                (hdd/automaton-source
                  (hdd/difference rocks-domain bread-domain)
                  (hdd/clj->vsa* {:thin? true})
                  (hdd/clj->vsa* {:scrub-off butter}))))
'(:rocks)

;; (compare with intersection)
(hdd/cleanup*
 (recover
  item-memory
  (hdd/automaton-source
   (hdd/intersection-1 [rocks-domain bread-domain])
   (hdd/clj->vsa* {:thin? true})
   (hdd/clj->vsa* {:scrub-off butter}))))
'(:bread :rocks)

(hdd/cleanup*
 (recover
  item-memory
  (hdd/automaton-source
   (hdd/intersection-1 [rocks-domain bread-domain])
   (hdd/clj->vsa* {:thin? true})
   (hdd/clj->vsa* {:scrub-off #{butter lava}}))))
'(:bread :rocks)


;; with union
(hdd/cleanup*
 (recover
  item-memory
  (hdd/automaton-source
   (hdd/union rocks-domain bread-domain)
   (hdd/clj->vsa* {:thin? true})
   (hdd/clj->vsa* {:scrub-off butter}))))
'(:bread)
;; then bread, guess thatt is a coinflip with the item memory or sth in that case
;;

;; quering with both lava and butter
(hdd/cleanup*
 (recover
  item-memory
  (hdd/automaton-source
   (hdd/union rocks-domain bread-domain)
   (hdd/clj->vsa* {:thin? true})
   (hdd/clj->vsa* {:scrub-off #{butter lava}}))))
'(:rocks)
;; now rocks

;; rocks or nothing
(for [n (range 5)]
  (hdd/cleanup* (recover
                 item-memory
                 (hdd/automaton-source
                  (hdd/difference rocks-domain bread-domain)
                  (hdd/clj->vsa* {:thin? true})
                  (hdd/clj->vsa* {:scrub-off (hd/->seed)})))))
'((:rocks) (:rocks) (:rocks) (:rocks) ())


;; bread or nothing
;; guess that is random
(for [n (range 5)]
  (some->
   (recover
    item-memory
    (hdd/automaton-source
     ;; with intersection instead
     (hdd/intersection-1 [rocks-domain bread-domain])
     (hdd/clj->vsa* {:thin? true})
     (hdd/clj->vsa* {:scrub-off (hd/->seed)})))
   hdd/cleanup*))
'(nil (:bread) (:bread) nil nil)











;; -------------------------

(comment
  (remember-leaves!
   (hdd/clj->vsa*
    [[{:bread {:surface :empty}} {:spread butter}
      {:bread {:surface butter}}]
     [{:bread {:surface butter}} {:spread butter}
      {:bread {:surface butter :thick? true}}]
     [{:bread {:surface butter}} {:scrub-off butter}
      {:bread {:surface butter :thin? true}}]]))

  (hd/similarity
   (hdd/clj->vsa* {:bread {:surface :empty}})
   (recover item-memory (hdd/clj->vsa* {:bread {:surface butter}})))
  0.0
  (hd/similarity
   (hdd/clj->vsa* {:bread {:surface butter}})
   (recover item-memory (hdd/clj->vsa* {:bread {:surface butter}})))
  1.0)

(comment
  (do (remember item-memory
                (hdd/clj->vsa* #{:bread {:surface :empty}}))
      (= (hdd/clj->vsa* #{:bread {:surface :empty}})
         (recover item-memory
                  (hdd/clj->vsa* #{:bread
                                   {:surface :empty}}))))
  true
  (do
    (doseq [x (hdv-nodes
               (hdd/clj->vsa*
                [[#{:bread {:surface :empty}}
                  #{:spread :butter}
                  #{:bread {:surface :butter}}]
                 [#{:bread {:surface :butter}}
                  #{:spread :butter}
                  #{:bread
                    {:surface :butter :thick? true}}]
                 [#{:bread {:surface :butter}}
                  #{:scrub-off :butter}
                  #{:bread
                    {:surface :butter :thin? true}}]]))]
      (remember item-memory x)
      (remember item-memory
                (hdd/clj->vsa* #{{:surface :empty} :bread}))
      (remember item-memory
                (hdd/clj->vsa* #{{:surface :empty} :bread}))
      (remember item-memory
                (hdd/clj->vsa* #{{:surface :empty}
                                 :bread})))
    (let [bread1outcome (recover item-memory
                                 (hdd/clj->vsa*
                                  #{:bread
                                    {:surface :empty}}))]
      (map (fn [symbolic-item]
             {:bread1outcome-sum (f/sum bread1outcome)
              :equal? (= bread1outcome
                         (hdd/clj->vsa* symbolic-item))
              :hdv-sum (f/sum (hdd/clj->vsa* symbolic-item))
              :similarity (hd/similarity bread1outcome
                                         (hdd/clj->vsa*
                                          symbolic-item))
              :symbolic-item symbolic-item})
           [#{:bread {:surface :empty}} #{:spread :butter}
            #{:bread {:surface :butter}}
            #{:bread {:surface :butter}} #{:spread :butter}
            #{:bread {:surface :butter :thick? true}}
            #{:bread {:surface :butter}} #{:scrub-off :butter}
            #{:bread {:surface :butter :thin? true}}]))))

















;; ---------------------------------------------------------------------




;; permuting sums:
;; e follows d
(let [d (->prototype :d)
      e (->prototype :e)
      ;; you can't thin then
      s (hd/bundle (hd/permute d) e)
      s2 (->store-item s)]
  ;; ... we encountered d
  ;; ask what follows d?
  (= s
     (cleanup-lookup-value
      ;; cleanup-lookup-value
      (hd/permute d)))
  (cleanup-lookup-value
   (f/- s (hd/permute d))))

;; yea, this you cannot do with the thinning
(let [d (->prototype :d)
      e (->prototype :e)
      ;; you can't thin then
      s (hd/thin (hd/bundle (hd/permute d) e))
      s2 (->store-item s)]
  ;; ... we encountered d
  ;; ask what follows d?
  (= s
     (cleanup-lookup-value
      ;; cleanup-lookup-value
      (hd/permute d)))
  (cleanup-lookup-value
   (f/- s (hd/permute d))))



(let [d (->prototype :d)
      e (->prototype :e)
      f (->prototype :f)
      s (hd/thin
         (hd/bundle
          (hd/permute d)
          (hd/bind (hd/permute d) e)
          (hd/bind (hd/permute e) f)
          (hd/bind (hd/permute-n d 2) f)
          ))
      s2 (->store-item s)]

  (= s
     (cleanup-lookup-value
      (hd/permute d)))

  ;; ... we encountered d
  ;; ask what follows d?
  ;; (cleanup-lookup-value (hd/unbind (cleanup-lookup-value
  ;;                                   (hd/permute d))
  ;;                                  (hd/permute d)))

  [(cleanup-lookup-value
    (hd/unbind
     (cleanup-lookup-value (hd/permute d))
     (hd/permute-n d 2)))
   (cleanup-lookup-value
    (hd/unbind
     (cleanup-lookup-value (hd/permute d))
     (hd/permute e)))])


(let [d (->prototype :d)
      e (->prototype :e)
      f (->prototype :f)
      s (hd/thin (hd/bundle
                  (hd/permute d)
                  (hd/bind (hd/permute d) e)
                  (hd/bind (hd/permute-n d 2) f)))
      s2 (->store-item s)]
  (= s (cleanup-lookup-value (hd/permute d)))
  ;; ... we encountered d
  ;; ask what follows d?
  ;; (cleanup-lookup-value (hd/unbind
  ;; (cleanup-lookup-value
  ;;                                   (hd/permute d))
  ;;                                  (hd/permute d)))
  [(cleanup-lookup-value d)
   (cleanup-lookup-value (hd/unbind (cleanup-lookup-value
                                     (hd/permute d))
                                    (hd/permute-n d 1)))
   (cleanup-lookup-value (hd/unbind (cleanup-lookup-value
                                     (hd/permute d))
                                    (hd/permute-n d 2)))])



(let [seq (map ->prototype (map #(* 10 %) (range 3)))
      s
      (apply
       hd/bundle
       (concat
        ;; this is my seq handle. Like
        ;; when you start singing 'abc...'
        [(hd/permute (first seq))]
        (map-indexed
         (fn [idx [a b]]
           (hd/bind (hd/permute a)
                    (hd/permute-n (first seq) (inc idx))))
         (map vector seq (next seq)))))
      s2 (->store-item s)]
  ;; ... we encountered d
  ;; ask what follows d?
  ;; (cleanup-lookup-value (hd/unbind
  ;; (cleanup-lookup-value
  ;;                                   (hd/permute d))
  ;;                                  (hd/permute d)))
  ;; [(cleanup-lookup-value d)
  ;;  (cleanup-lookup-value (hd/unbind
  ;;  (cleanup-lookup-value
  ;;                                    (hd/permute d))
  ;;                                   (hd/permute-n d
  ;;                                   1)))
  ;;  (cleanup-lookup-value (hd/unbind
  ;;  (cleanup-lookup-value
  ;;                                    (hd/permute d))
  ;;                                   (hd/permute-n d
  ;;                                   2)))]
  (hd/similarity s (hd/permute (first seq)))

  ;; (=
  ;;  s
  ;;  (cleanup-lookup-value (hd/permute (first seq))))

  (reduce
   (fn [acc idx]
     (if-let
         [v
          (cleanup-lookup-value
           (hd/unbind
            s
            (hd/permute-n
             (first seq)
             (inc idx))))]
         (conj acc)
         (ensure-reduced acc)))
   ;; (cleanup-lookup-value (hd/permute (first seq)))
   []
   (range)))

;; -> this doesn't work so well :P
















(let [d (->prototype :d)
      e (->prototype :e)
      pair (->record {:first :d :second :e})]
  (cleanup-lookup-value
   (hd/unbind pair (->prototype :first))))










(do
  (def a (hd/->hv))
  (def b (hd/->hv))
  (def c (hd/->hv))

  (->sequence [0 1 2 3])

  (hd/bundle a (hd/permute b))

  (let [a (->prototype 0)
        b (->prototype 1)
        c (->prototype 2)]
    ;; (hd/bind
    ;;  a
    ;;  (hd/permute b))

    (hd/thin
     (hd/bundle
      a
      (hd/permute-n b 1)
      (hd/permute-n c 2)))))


(let [a (->prototype 0)
      b (->prototype 1)
      c (->prototype 2)
      r
      ;; (hd/bind
      ;;  a
      ;;  (hd/permute b))
      (hd/thin
       (hd/bundle
        a
        (hd/permute-n b 1)
        (hd/permute-n c 2)))]


  (cleanup-lookup-value r)

  (cleanup-lookup-value
   (hd/permute-n
    (hd/unbind
     r
     (->prototype (cleanup-lookup-value r)))
    -1))


  (cleanup-lookup-value
   (hd/permute-inverse
    (hd/unbind
     r
     (->prototype (cleanup-lookup-value r))))))


(let [a (->prototype 0)
      b (->prototype 1)
      c (->prototype 2)
      r
      ;; (hd/bind
      ;;  a
      ;;  (hd/permute b))
      (hd/thin
       (hd/bundle a (hd/permute-n b 1)))]
  (hd/similarity
   (->prototype (cleanup-lookup-value r))
   (hd/permute-n b 1))


  ;; (cleanup-lookup-value
  ;;  (hd/permute-inverse
  ;;   (hd/unbind
  ;;    r
  ;;    (->prototype (cleanup-lookup-value r)))))
  )


(let [a (->prototype 0)
      b (->prototype 1)
      c (->prototype 2)
      r
      (hd/bind a (hd/permute-n b 1))]

  (cleanup-lookup-value
   (hd/permute-inverse
    (hd/unbind
     r
     (->prototype (cleanup-lookup-value r))))))


;; ===========================================================



(let [a (hd/->hv)
      noise (apply f/+ (repeatedly 20 hd/->hv))
      a' (hd/thin (f/+ a noise))]
  (hd/similarity a a'))

(apply
 max
 (let
     [a (hd/->hv)]
     (for [n (range 50000)]
       (hd/similarity a (hd/->hv)))))



(defn predicate-branches [predicate-value]
  (cleanup-lookup-verbose predicate-value)
  )

(defmacro hyper-if
  [predicate consequence alternative]
  `(let
       [condition ~condition]
       (if condition ~consequence ~alternative)))

(some odd? [1 2 3 4 5 6 7 8 9 10])

(let [fluid (->prototype :a)
      water (->prototype :b)])

(let [fluid (->prototype :a)
      water (->prototype :b)
      lava (->prototype :c)
      water (hd/thin (hd/bundle fluid water))
      lava (hd/thin (hd/bundle fluid lava))
      fire (->prototype :d)
      lava (hd/thin (hd/bundle lava fire))]
  [
   (hd/similarity fluid fire)
   (hd/similarity fluid lava)
   (hd/similarity fluid water)
   (hd/similarity water lava)])



(let [a (hd/->hv)
      b (hd/->hv)
      random-and-similar-to-both
      (fn []
        (let [c (hd/->hv)
              c (hd/thin (hd/bundle a b c))]
          c))
      similar? (fn [similarity] (< 0.09 similarity))]
  (hd/similarity (random-and-similar-to-both)
                 (random-and-similar-to-both))
  ;; [[:a :b (similar? (hd/similarity a b))]
  ;;  [:a :random-c-min
  ;;   ;; make a 100 random vectors, look at the min
  ;;   ;; similarity
  ;;   (similar?
  ;;    (apply
  ;;     min
  ;;     (map (fn [_]
  ;;            (hd/similarity a (random-and-similar-to-both)))
  ;;          (range 100))))]
  ;;  [:b :random-c-min
  ;;   (similar?
  ;;    (apply min
  ;;           (map (fn [_]
  ;;                  (hd/similarity b (random-and-similar-to-both)))
  ;;                (range 100))))]]
  )


[[:a :b false]
 [:a :random-c-min true]
 [:b :random-c-min true]]



;; Make a heteroassociative memory as bridge to clj
;;
;;
;; Then
;;
;; (lookup cam a)
;; -> a
;;
;; (lookup am a)
;; -> clj-data
;;
;; next(a):
;; (lookup cam (unbind a (permute a))
;;

(def h-memory (atom {}))


(defn ->record [kvps]
  (hd/thin
   (apply
    hd/bundle
    (for [[k v] kvps]
      (hd/bind
       (->prototype k)
       (->prototype v))))))



(def sequence-token (->prototype :sequence))

(defn sequence? [o]
  (lookup cam (hd/bind o sequence-token)))


(defn ->sequence
  [xs]
  ;; version 1: Pointer chains that would be
  ;; heteroassociative chain can start anywhere, but if
  ;; the items comes up twice, it collides
  ;;
  ;;
  ;; A sequence is now the same thing as it's first
  ;; item.
  ;;
  ;; This poses the problem of multiple sequences with
  ;; the same start. Also, sequences with the same item
  ;; at pos n, would not be differentiated
  ;;
  ;; A solution to these problems might be 'context'.
  ;; The caller can implement this by making `a` a
  ;; gensym,. When thy construct the sequence, they can
  ;; bind all vectors with `a`. Then when they consume
  ;; the sequence, they can unbind with `a`.
  ;;
  ;; If the processor is updating the context while it
  ;; unrolls the sequence, it would be able to branch
  ;; depending on the context.
  ;;
  ;;
  ;;
  ;; Q is the identity of the sequence
  ;; Map all elements to Q-space,. This protects the
  ;; memory of different sequences
  (let [Q (hd/->hv)
        xs (concat [Q] xs)
        xs (map (fn [x] (hd/bind x Q)) xs)]
    (doall
     (map (fn [a b]
            (let [a->b (hd/bind a (hd/permute a))]
              ;; [ a . ]
              (store cam a)
              ;; [ . b ]
              (store cam a->b)
              (swap! h-memory assoc a->b b)
              a))
          xs (next xs)))
    Q))



(=
 (->sequence [(->prototype :a) (->prototype :b) (->prototype :c)])
 (->sequence [(->prototype :a) (->prototype :b) (->prototype :c)]))





(let [a (->prototype :a)]
  (= a
     (hd/unbind (hd/bind a (hd/permute a))
                (hd/permute a))))

(comment
  (let [seq-handle (->sequence [(->prototype :a)
                                (->prototype :b)
                                (->prototype :c)])]
    (take-while
     identity
     (iterate (fn [a]
                (when a
                  (let [a (lookup cam a)]
                    (when a
                      (let [a->b (hd/bind a (hd/permute a))
                            b->b (lookup cam a->b)]
                        (@h-memory a->b))))))
              (lookup cam seq-handle)))))




;; returns a potentionally infinite sequence of the next items in the sequence
(defn seq-read
  [seq-identifier]
  (let [first-item (@h-memory seq-identifier)]
    (take-while
      identity
      (iterate
        (fn [a]
          (when a
            (let [a (lookup cam a)]
              (when a
                (let [a->b (hd/bind a (hd/permute a))
                      a->b (lookup cam a->b)]
                  (when a->b
                    (if (sequence? a->b)
                      (seq-read a->b)
                      (when-let [b (@h-memory a->b)]
                        (hd/unbind b seq-identifier)))))))))
        first-item))))


(seq-read
 (->sequence
  [(->prototype :a)
   (->prototype :b)
   (->prototype :c)]))





(map
 cleanup-lookup-value
  (let [Q
        (->sequence [(->prototype :a)
                     (->prototype :b)

                     (->prototype :c)

                     ])
        seq-first (lookup cam (hd/bind Q Q))]
    Q
    (map (fn [x] (hd/unbind x Q))
      (take-while
        identity
        (drop
         1
         (iterate (fn [a]
                    (when a
                      (let [a (lookup cam a)]
                        (when a
                          (let [a->b (hd/bind a
                                              (hd/permute a))
                                b->b (lookup cam a->b)]
                            (@h-memory a->b))))))
                  seq-first))))))





(defn seq-unroll
  [Q]
  (let [seq-first (lookup cam (hd/bind Q Q))]
    (map
     (fn [x] (hd/unbind x Q))
     (take-while
      identity
      (drop
       1
       (iterate
        (fn [a]
          (when a
            (let [a (lookup cam a)]
              (when a
                (let [a->b (hd/bind a (hd/permute a))
                      b->b (lookup cam a->b)
                      b (@h-memory a->b)]
                  (when b
                    (or (let [b (hd/unbind b Q)]
                          (when (lookup cam (hd/bind b b))
                            (seq-unroll b)))
                        b)))))))
        seq-first))))))


(map
 cleanup-lookup-value
 (let [Q (->sequence [(->prototype :a)
                      ;; (->prototype :b)
                      (->sequence [(->prototype :a)
                                   (->prototype :b)
                                   ;; (->prototype :c)
                                   ])
                      ;; (->prototype :c)
                      ])]
   (seq-unroll Q)))





(let [Q (hd/->hv)
      a (->prototype :a)
      b (->prototype :b)
      xs (map (fn [x] (hd/bind Q x)) [a b])
      first-item
      (hd/bind (first xs) (hd/permute (first xs)))
      ]
  (hd/bind a (hd/permute a))
  (hd/similarity
   (hd/bind Q a)
   (first xs)))





;; quickly do the boring thing and make a hdv list processor

(defn clj->vsa
  [obj]
  (cond
    (map? obj) (->record obj)
    (vector? obj)
    (->sequence (map clj->vsa obj))
    :else (->prototype obj)))

(comment
  (into []
        (map
         cleanup-lookup-value
         (seq-next (clj->vsa [:a :b :c]))))
  [:a :b :c]

  (into []
        (map
         cleanup-lookup-value
         (seq-next
          (clj->vsa [:a
                     [:b :d :e :f]
                     :c]))))
  (into []
        (map
         cleanup-lookup-value
         (seq-next
          (clj->vsa
           [:b :d :e :f]))))
  [:b :d :e :f]


  )


(def lambda-token (->prototype :λ))
(def if-token (->prototype :if))

(defn primitive-op? [o]
  (ifn? o))

(defn lambda? [o]
  (= :λ (cleanup-lookup-value o)))

(defn if? [o]
  (= :if (cleanup-lookup-value o)))

(defn sp-eval [hv]
  (cond
    )

  )


(defn sp-apply [op arguments]
  (cond
    (primitive-op? op)
    (apply op arguments)



    ))



;; not like this
(comment

  (let [letexp (clj->vsa '(let [a 10 b 20] (+ a b)))
        exp letexp
        bindings
        (for [[k v]
              (partition
               2
               (unroll (known (h-nth exp 1))))]
          [k (h-eval v)])
        body (h-nth exp 2)
        new-env
        (if bindings
          (hd/thin
           (reduce
            (fn [env [k v]]
              (augment-environment
               env
               k
               (h-eval v)))
            (or *h-environment* (hd/->hv))
            bindings))
          *h-environment*)]
    ;; (eval-let letexp)
    (cleanup* (lookup-variable (clj->vsa 'b) new-env))
    (cleanup*
     (hd/unbind new-env (clj->vsa 'a))))


  (unroll
   (clj->vsa ['let '[a 10 b 20] [+ 'a 'b]]))

  (let [letexp (clj->vsa ['let '[a 10 b 20] [+ 'a 'b]])
        exp letexp
        bindings
        (for [[k v]
              (partition
               2
               (unroll (known (h-nth exp 1))))]
          [k (h-eval v)])
        body (h-nth exp 2)
        new-env
        (if bindings
          (hd/thin
           (reduce
            (fn [env [k v]]
              (augment-environment
               env
               k
               (h-eval v)))
            (or *h-environment* (hd/->hv))
            bindings))
          *h-environment*)]
    (cleanup*
     (hd/unbind new-env (clj->vsa 'a)))

    (binding [*h-environment* new-env]
      (cleanup*
       (h-eval (clj->vsa 'a))))

    (binding [*h-environment* new-env]
      ;; (map cleanup-lookup-value (unroll body))
      ;; (h-seq? body)
      ;; [(h-if? body) (let? body)]
      ;; (let
      ;;     [exp body]
      ;;     (let [lst (unroll exp)]
      ;;       ;; (h-apply
      ;;       ;;  (h-eval (first lst))
      ;;       ;;  (map h-eval (rest lst)))
      ;;       ;; (map cleanup-lookup-value (map h-eval (rest lst)))
      ;;       (map variable? (rest lst))
      ;;       (map h-eval (map known (rest lst)))
      ;;       (map cleanup*
      ;;            (map h-eval (map known (rest lst))))
      ;;       (map cleanup-lookup-value (rest lst))
      ;;       (doall (map cleanup-lookup-value (map h-eval (map clj->vsa '(a b)))))))

      (cleanup* (h-eval body))))


  '(30)

  ;; (false true true)

  ;; (10)

  (cleanup* (h-eval
             (clj->vsa ['let '[a 10 b 20] [+ 'a 'b]])))
  '(30)

  (cleanup*
   (h-eval
    (clj->vsa
     ['let '[a 10 b 20]
      ['let '[b 100]
       [+ 'a 'b]
       ]])))
  ;; (30 110)

  (cleanup*
   (h-eval
    (clj->vsa
     ['let '[a 10 b 20]
      ['let '[b 200]
       'b]])))
  ;; (200 20)


  (h-eval (clj->vsa ['let
                     ['a 10 'b [+ 200 600]]
                     [+ 'a 'b]]))
  ;; #tech.v3.tensor<int8>[10000]
  ;; [0 0 0 ... 0 0 0]
  ;; (cleanup* *1)
  ;; (810)
  (mix (hd/->hv) (hd/->hv))


  (let [a-quoted (hd/permute (clj->vsa 'a))]
    (= a-quoted (h-eval a-quoted)))



  (binding
      [*h-environment*
       (-> (augment-environment (hd/->hv)
                                (clj->vsa 'a)
                                (clj->vsa 10))
           (augment-environment (clj->vsa 'b)
                                (clj->vsa 20)))]
      (let [a-quoted (hd/permute (clj->vsa 'a))
            a (clj->vsa 'a)]
        [(cleanup* (h-eval a-quoted))
         (cleanup* (h-eval a))]))
  ;; [() (10)]


  (cleanup*
   (h-eval
    (clj->vsa
     '(let [a 10
            b 20]
        (+ a b)))))

  (cleanup* exp)

  (cleanup* (hd/unbind env (hd/permute-inverse exp))))

(comment

  ;; replace lava for water in a container record
  (let [container-lava
        (clj->vsa {:inside :lava :kind :bucket})
        water (clj->vsa :water)
        container-water
        (hd/bundle
         container-lava
         (hd/bind
          (hd/unbind container-lava (->prototype :lava))
          water))]
    [(cleanup*
      (hd/bind container-water
               (hd/inverse (->prototype :inside))))
     (cleanup*
      (hd/bind container-water
               (hd/inverse (->prototype :kind))))])


  ;; representing substitution

  (let [container-lava
        (clj->vsa {:inside :lava :kind :bucket})
        water (clj->vsa :water)

        ;; substitute :water with whatever :lava is

        ;; => :inside
        the-variable (hd/unbind container-lava (clj->vsa :lava))

        ;; [:inside :water]
        new-pair (hd/bind the-variable water)

        substitution
        (hd/bind (clj->vsa :lava) container-lava)]

    (cleanup*
     (hd/unbind
      (hd/bind (clj->vsa :lava) container-lava)
      container-lava)))


  (let [container-lava
        (clj->vsa {:inside :lava :kind :bucket})
        water (clj->vsa :water)

        ;; substitute :water with whatever :lava is

        ;; => :inside
        the-variable (hd/unbind container-lava (clj->vsa :lava))

        ;; [:inside :water]
        new-pair (hd/bind the-variable water)

        substitution
        (hd/bind (clj->vsa :lava) container-lava)]

    (cleanup*
     (hd/unbind
      (hd/bind (clj->vsa :lava) container-lava)
      container-lava))

    (cleanup*
     (f/-
      container-lava

      ;; :lava
      (hd/unbind container-lava (clj->vsa :inside)))))


  )

(comment
  (cleanup-lookup-value
   (lookup-variable
    (clj->vsa 'a)
    (fabricate-environment
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

(comment
  (f/bit-and
   (hd/->seed)
   (dtt/->tensor (concat (repeatedly (/ 1e4 5)) (repeatedly))))


  (defn select-k-segment
    [v k]
    (f/bit-and v
               (dtt/compute-tensor
                [word-length]
                (fn [i]
                  (< (* (/ word-length 5) k)
                     i
                     (* (/ word-length 5) (inc k)))))))


  (f/bit-and [1 0 0] (dtt/compute-tensor [3] (fn [i] true)))


  (let
      [a (hd/->seed)
       a1 (select-k-segment a 0)
       a2 (select-k-segment a 1)
       b (hd/->seed)
       b1 (select-k-segment b 0)]
    ;; [ label-segment L=2.000, ...  ]
    ;; similarity of 0.2 then would be complete overlap
      (hd/similarity a1 a)
      ;; then interestingly, mixing in more segments makes
      ;; the label overlap stay
      ;;
      (hd/similarity (f/+ a1 b1) a)
      ;; and of course that would then be similar to what is mixed
      (hd/similarity (f/+ a1 b1 a2) a)))


;; --
;; another explorative attempt at a resonator

;; ----

(def codebooks
  [(sdm/->sdm
    {:address-count (long 1e6)
     :address-density 0.000003
     :word-length (long 1e4)})
   (sdm/->sdm
    {:address-count (long 1e6)
     :address-density 0.000003
     :word-length (long 1e4)})
   (sdm/->sdm
    {:address-count (long 1e6)
     :address-density 0.000003
     :word-length (long 1e4)})])

(doall
 (map
  (fn [factors sdm]
    (doseq [f factors]
      (sdm/write sdm f f 1)))
  (map-indexed
   (fn [i factors] (map (fn [x] (hd/permute-n x i)) factors))
   (hdd/clj->vsa*
    [[:s0 :s1]
     [0 1 :halt false true]
     [:right :left :-]]))
  codebooks))

;; -------------

(def x (hdd/clj->vsa* [:*> :s0 0 :right]))

(defn bounce-resonator
  [codebooks x]
  (reductions
    (fn [{:keys [best-guesses confidence excitability]} n]
      (let [new-confidence
              (hd/similarity (hd/bind* best-guesses) x)
            excitability (max 5
                              (min 1
                                   (if (<= confidence
                                           new-confidence)
                                     (inc excitability)
                                     (dec excitability))))]
        (if (<= 0.99 confidence)
          (ensure-reduced {:best-guesses best-guesses
                           :confidence confidence
                           :n n})
          {:best-guesses (into []
                               (map (fn [sdm guess]
                                      (pyutils/torch->jvm
                                        (:result
                                          (sdm/lookup
                                            sdm
                                            guess
                                            excitability
                                            1))))
                                 codebooks
                                 best-guesses))
           :confidence confidence
           :n n})))
    {:best-guesses (for [sdm codebooks]
                     (pyutils/torch->jvm
                       (:result
                         (sdm/lookup sdm (hd/->ones) 4 1))))
     :confidence 0
     :excitability 4}
    (range 10)))

(bounce-resonator codebooks x)


(def best-guesses
  (into []
        (for [sdm codebooks]
          (pyutils/torch->jvm
           (:result (sdm/lookup sdm (hd/->ones) 4 1))))))

(map hdd/cleanup* (map-indexed (fn [i x] (hd/permute-n x (- i))) best-guesses))


'((:s0 :s1) (0 :halt true false) (:right :- :left))
'((:s0 :s1) (:halt true false) (:right :- :left))
'((:s0 :s1) (0 :halt true false) (:right :- :left))

(hdd/cleanup* (hd/unbind x (hd/bind* (rest best-guesses))))

(hd/similarity
 (hdd/clj->vsa* [:+ :s0 :s1])
 (hd/unbind x (hdd/clj->vsa*
               [:*
                [:> [:+ 0 1 true false]]
                [:>> [:+ :- :right :right]]])))

(hd/similarity
  (hd/bind* (into []
                  (for [sdm codebooks]
                    (pyutils/torch->jvm
                      (:result (sdm/lookup sdm
                                           (hd/drop-randomly
                                             (hd/->ones)
                                             0.5)
                                           1
                                           1))))))
  x)




(hd/similarity x (hd/bind* best-guesses))

(hd/similarity x
               (hdd/clj->vsa*
                [:*>
                 [:+ :s0 :s1]
                 [:+ 0 1 true false]
                 [:+ :- :right :right]]))

(hdd/cleanup* (first best-guesses))
'(:s0 :s1)
(hdd/cleanup* (hd/permute-inverse (second best-guesses)))
(hdd/cleanup* (hd/permute-inverse (hd/permute-inverse (second (rest best-guesses)))))
