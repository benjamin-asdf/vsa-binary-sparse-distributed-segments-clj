
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
