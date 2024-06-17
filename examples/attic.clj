
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
      (hd/unbind container-lava (clj->vsa :inside))))))
