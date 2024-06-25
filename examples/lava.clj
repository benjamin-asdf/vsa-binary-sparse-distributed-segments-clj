

(comment



  #_(def pour
      (λ [{:keys [inside]} other-container]
         (substitue inside other-container)))


  ;; dirty pour

  (def pour1
    (fn [container1 container2]
      (hd/thin
       (hd/bundle
        container2
        (hd/bind
         (clj->vsa :inside)
         (hd/unbind container1 (clj->vsa :inside)))))))


  (let [my-lava-bucket (pour1 (clj->vsa {:inside :lava})
                              (clj->vsa {:bucket? true
                                         :inside :empty}))]
    {:bucket? (cleanup* (hd/unbind my-lava-bucket
                                   (clj->vsa :bucket?)))
     :inside (cleanup* (hd/unbind my-lava-bucket
                                  (clj->vsa :inside)))})

  ;; => {:bucket? (true) :inside (:lava :empty)}

  ;; substitution pour
  (def pour2
    (fn [container1 container2]
      (let [container2-inside
            (known
             (hd/unbind container2
                        (clj->vsa :inside)))]
        (hd/thin (hd/bundle
                  ;; turns out you can dissoc from a sumset by substracting with the kvp
                  ;; You might want to do this with cleaned up vecs, or decide some messy ness is good.
                  ;;
                  (f/- container2
                       (hd/bind (clj->vsa :inside)
                                container2-inside))
                  (hd/bind (clj->vsa :inside)
                           (hd/unbind
                            container1
                            (clj->vsa
                             :inside))))))))


  (let [my-lava-bucket (pour2 (clj->vsa {:inside :lava})
                              (clj->vsa {:bucket? true
                                         :inside :empty}))]
    {:bucket? (cleanup* (hd/unbind my-lava-bucket (clj->vsa :bucket?)))
     :inside (cleanup-lookup-verbose (hd/unbind my-lava-bucket
                                                (clj->vsa :inside)))})


  ;; {:bucket? (true)
  ;;  :inside
  ;;  ({:k :lava
  ;;    :similarity 0.76
  ;;    :v #tech.v3.tensor<int8> [10000]
  ;;    [0 0 0 ... 0 0 0]})}


  ;; that's a bucket with both lava and water
  (let [lava-bucket (clj->vsa {:bucket? true :inside :lava})
        water-bucket (clj->vsa {:bucket? true :inside :water})
        super-bucket (hd/thin (hd/bundle lava-bucket
                                         water-bucket))]
    {:bucket? (cleanup* (hd/unbind super-bucket
                                   (clj->vsa :bucket?)))
     :inside (cleanup* (hd/unbind super-bucket
                                  (clj->vsa :inside)))})


  ;; {:bucket? (true), :inside (:lava :water)}


  ;; dirty subst
  (def pour3
    (fn [container1 container2]
      (let [container2-inside (hd/unbind container2
                                         (clj->vsa :inside))]
        (hd/thin (hd/bundle (f/- container2
                                 (hd/bind (clj->vsa :inside)
                                          container2-inside))
                            (hd/bind (clj->vsa :inside)
                                     (hd/unbind
                                      container1
                                      (clj->vsa
                                       :inside))))))))

  (let [lava-bucket (clj->vsa {:bucket? true :inside :lava})
        water-bucket (clj->vsa {:bucket? true :inside :water})
        super-bucket (hd/thin (hd/bundle lava-bucket
                                         water-bucket))
        super-bucket
        (pour3 lava-bucket super-bucket)]

    {:bucket? (cleanup* (hd/unbind super-bucket
                                   (clj->vsa :bucket?)))
     :inside (cleanup* (hd/unbind super-bucket
                                  (clj->vsa :inside)))})

  ;; {:bucket? (true), :inside (:lava)}

  (def inside (fn [a] (hd/unbind a (clj->vsa :inside))))

  ;; just an idea

  (defn fulfills-role? [filler role]
    (known
     (hd/bind role (hd/permute filler))))

  (defn assign-role [filler role]
    (remember-soft (hd/bind role (hd/permute filler))))

  (def spread
    (fn [bread-like butter-like]
      (if-not (fulfills-role? butter-like (clj->vsa :butter))
        ;; 'nonsense'
        (hd/->hv)
        ;; that's just a conj
        (hd/thin (hd/bundle bread-like
                            (hd/bind (clj->vsa :butter)
                                     butter-like))))))


  ;; it's vegan butter btw
  (let [butter-prototype (clj->vsa :butter)
        _ (assign-role (clj->vsa :nectar) (clj->vsa :butter))
        _ (assign-role (clj->vsa :lava) (clj->vsa :butter))]
    (cleanup*
     (hd/unbind
      (spread
       (clj->vsa {:bread? true})
       (clj->vsa :nectar))
      (clj->vsa :butter))))
  ;; (:nectar)

  (let [butter-prototype (clj->vsa :butter)
        _ (assign-role (clj->vsa :nectar) (clj->vsa :butter))
        _ (assign-role (clj->vsa :lava) (clj->vsa :butter))]
    (cleanup*
     (hd/unbind
      (spread
       (clj->vsa {:bread? true})
       (clj->vsa :rocks))
      (clj->vsa :butter))))
  ;; ()


  (let [butter-prototype (clj->vsa :butter)
        _ (assign-role (clj->vsa :nectar) (clj->vsa :butter))
        _ (assign-role (clj->vsa :lava) (clj->vsa :butter))]
    (cleanup*
     (hd/unbind
      (spread
       (clj->vsa {:bread? true})
       (mix1 :nectar :lava))
      (clj->vsa :butter))))

  ;; the superposition of nectar and lava happen to be butter-like
  ;; (because it finds one of them to fulfill the role)

  ;; (:lava :nectar)

  (let [butter-prototype (clj->vsa :butter)
        _ (assign-role (clj->vsa :nectar) (clj->vsa :butter))
        _ (assign-role (clj->vsa :lava) (clj->vsa :butter))]
    (cleanup*
     (hd/unbind
      (spread
       (clj->vsa {:bread? true})
       (mix1 :nectar :rocks))
      (clj->vsa :butter))))

  ;; ... rocks nectar would also work
  ;; (:rocks :nectar)
  ;;
  ;; now I imagine nectar with little rocks on the bread.
  ;; it's crunchy
  ;;
  ;;

  ;; ... unless maybe you select whatever fits the butter role out of the superposition?
  ;; (but I leave it)



  ;; --------------------------------
  ;; lava and water are not merely associated
  ;; perhaps they should be *the same*, given the right context
  )
