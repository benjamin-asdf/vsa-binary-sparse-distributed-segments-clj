
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
