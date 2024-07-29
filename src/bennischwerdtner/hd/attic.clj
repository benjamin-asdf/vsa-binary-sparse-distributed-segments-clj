(comment
  (cleanup (peek (stack (clj->vsa :a) (clj->vsa :b))))
  (cleanup (pop (stack (clj->vsa :a) (clj->vsa :b))))
  (cleanup (hd/permute (pop (stack (clj->vsa :a) (clj->vsa :b)))))
  (cleanup (pop-clean (stack (clj->vsa :a) (clj->vsa :b)) (fn [x] (clj->vsa (cleanup x)))))


  ;; ... and now you can pop 'segment-length' often, then you are back to the original stack
  ;; this is the main downside to not eliminating the item from the stack in the pop operation
  ;; (the other being that the hdv get's denser)
  ;;

  ;;
  ;; segment-length is 500 here
  ;;
  (keep cleanup (take 550 (iterate pop (stack (clj->vsa :a) (clj->vsa :b)))))
  '(:a :b :a :b)
  ;; This gives an upper bound for the depth for such a stack. After 500 times, the first item would interfere with what you add
  ;; ... (I realzed that at more than 7 items the stack is too similar to everything anyway)
  ;;


  ;; not so with pop-clean
  (keep cleanup (take 550 (iterate #(pop-clean % (fn [x] (clj->vsa (cleanup x)))) (stack (clj->vsa :a) (clj->vsa :b)))))
  '(:a :b)

  ;; ---------------------------------------
  ;;

  ;; fill item memory with stuff
  (for [n (range 1000)]
    (clj->vsa n))
  (map
   cleanup*
   (reductions
    (fn [stack idx]
      (stack-conj stack (clj->vsa idx)))
    (stack (clj->vsa :a) (clj->vsa :b))
    (range 100)))

  (def stack1
    (last (reductions (fn [stack idx]
                        (hd/thin (stack-conj stack
                                             (clj->vsa idx))))
                      (stack (clj->vsa :a) (clj->vsa :b))
                      (range 100))))

  ;; this works not good with segment-length 500
  (keep cleanup (take 120 (iterate pop stack1)))

  ;; also with segment-length 100
  ;; I get like the last 4 items back
  ;;

  )



;;
;; the adventures of trying to make a directed bind
;; .. then I figured out that my permute had a misconception.
;; and a good permute solves my problems!
;;
(comment
  (defn directed-bind
    "
  "
    ([hdvs] (bind* hdvs default-opts))
    ([hdvs {:bsdc-seg/keys [segment-count segment-length]}]
     (indices->hv (dtype/emap
                   #(fm/mod % segment-length)
                   :int8
                   (let [indices (dtt/reduce-axis
                                  (map #(dtt/reshape
                                         %
                                         [segment-count
                                          segment-length])
                                       hdvs)
                                  dtype-argops/argmax)]
                     (map f/*
                          (dtt/rows indices)
                          (range 1 (inc (count indices)))))))))

  (defn directed-unbind [a b])



  (def a (->seed))
  (def b (->seed))
  ;; b is double as important as a
  (directed-bind [a b])

  (=
   (directed-bind [a b])
   (bind b a 2))

  89

  (= (bind a b 2) (bind a (bind a b)))
  (= b (unbind (unbind (bind a (bind a b)) a) a))

  (= b (unbind (unbind (bind a b 2) a) a))

  (= b (unbind (bind a (bind a b 2) -1) a))

  (= b (bind a (bind a b 2) -2))

  (= b (bind a (bind a b 2) -2))

  (let [e (bind a b 2)]
    (= a (indices->hv (f/* (hv->indices (unbind e b)) 1/2))))


  (hv->indices a)
  (hv->indices b)

  (dtype/emap
   #(fm/mod % 500)
   :int8
   (f/+
    (f/* (hv->indices a) 2)
    (hv->indices b)))
  (f/+ (f/* (hv->indices a) 2) (hv->indices b))

  [1050 909 350 602 967 827 994 1123 1087 531 503 547 980 626 631 781 303 130 961 1157]
  [50 409 350 102 467 327 494 123 87 31 3 47 480 126 131 281 303 130 461 157]

  (=
   (indices->hv (dtype/emap #(fm/mod % 500)
                            :int8
                            (f/+ (f/* (hv->indices a) 2)
                                 (hv->indices b))))
   (bind a b 2))

  (def c (bind a b 2))

  (indices->hv
   (dtype/emap #(fm/mod % 500) :int8 (f/+ (f/* (hv->indices a) 2) (hv->indices b))))

  (hv->indices a)
  #tech.v3.tensor<int8> [20]
  [482 410 151 194 422 306 424 404 353 248 101 174 453 276 220 141 35 58 422 356]
  (hv->indices b)
  #tech.v3.tensor<int8> [20]
  [86 89 48 214 123 215 146 315 381 35 301 199 74 74 191 499 233 14 117 445]

  (hv->indices c)
  #tech.v3.tensor<int8> [20]
  [50 409 350 102 467 327 494 123 87 31 3 47 480 126 131 281 303 130 461 157]

  (mod (+ 86 (* 2 482)) 500)
  50

  ;; other way around:
  (mod (* (- 50 86) 1/2) 500)

  (dtype/emap
   #(fm/mod % 500)
   :int8
   (dtt/->tensor
    (f/*
     (f/-
      (hv->indices c)
      (hv->indices b))
     1/2)))

  (similarity
   a
   (indices->hv (dtype/emap #(fm/mod % 500) :int8 (f/* (f/- (hv->indices c) (hv->indices b)) 0.5))))
  0.35

  (dtype/emap #(fm/mod % 500) :int8 (f/* (f/- (hv->indices c) (hv->indices b)) 0.5))
  [482.0 160.0 151.0 444.0 172.0 56.0 174.0 404.0 353.0 498.0 351.0 424.0 203.0 26.0 470.0 391.0 35.0 58.0 172.0 356.0]

  (mod (/ (- 409 89) 2) 500)

  ;; a*2 + b = c
  (mod (+ (* 2 410) 89) 500)
  409

  (mod (+ (* 2 (second (hv->indices a))) (second (hv->indices b))) 500)


  (f/+ (f/* (hv->indices a) 2) (hv->indices b))
  [1050 909 350 602 967 827 994 1123 1087 531 503 547 980 626 631 781 303 130 961 1157]

  (dtype/emap
   #(fm/mod % 500)
   :int8
   (dtt/->tensor [1050 909 350 602 967 827 994 1123
                  1087 531 503 547 980 626 631 781
                  303 130 961 1157]))

  (= c (indices->hv [50 409 350 102 467 327 494 123 87 31 3 47 480 126 131 281 303 130 461 157]))


  (hv->indices b)
  #tech.v3.tensor<int8> [20]
  [86 89 48 214 123 215 146 315 381 35 301 199 74 74 191 499 233 14 117 445]

  (f/- (hv->indices c) (hv->indices b))

  (/ (- 409 89) 2)

  [-36 320 302 -112 344 112 348 -192 -294 -4 -298 -152 406 52 -60 -218 70 116 344 -288]

  (second (hv->indices a))
  410




  ;; c
  (f/+ (f/* (hv->indices a) 2) (hv->indices b))

  ;; unbind
  (let [c (f/+ (f/* (hv->indices a) 2) (hv->indices b))]
    (= (hv->indices a) (f// (f/- c (hv->indices b)) 2)))

  (let [c (f/+ (f/* (hv->indices a) 2) (hv->indices b))]
    (= (hv->indices b) (f/- c (f/* (hv->indices a) 2))))



  (let [c (f/+ (f/* (hv->indices a) 2) (hv->indices b))
        c (map #(mod % 500) c)]
    (= (hv->indices b)
       (map #(mod % 500) (f/- c (f/* (hv->indices a) 2)))))


  ;; with this version unbinding for the source will only be approx the same
  ;; This is because with the mod, information is lost
  ;;

  (let [c (f/+ (f/* (hv->indices a) 2) (hv->indices b))
        c (map #(mod % 500) c)]
    ;; (similarity a)
    ;; (hv->indices)
    (similarity a
                (indices->hv (map #(mod % 500)
                                  (f// (f/- c (hv->indices b))
                                       2)))))

  ;; --------

  (let [c (f/+ (f/* (hv->indices a) (range 1 (inc 20))) (hv->indices b))
        c (map #(mod % 500) c)]
    (=
     (hv->indices b)
     (map #(mod % 500)
          (f/- c (f/* (hv->indices a)
                      (range 1 (inc 20)))))))


  (let [n (/ 1 3)
        c (f/+ (f/* (hv->indices a) n) (hv->indices b))
        c (map #(mod % 500) c)]
    c)

  (for [n [(/ 1 3) (/ 1 2)]]
    [n
     (let [c (f/+ (f/* (hv->indices a) n) (hv->indices b))
           c (map #(mod % 500) c)
           c-hv (indices->hv c)
           c2 (let [a b
                    b a]
                (map #(mod % 500)
                     (f/+ (f/* (hv->indices a) n)
                          (hv->indices b))))]
       [(similarity (indices->hv c) a)
        (similarity (indices->hv c) b)
        (similarity (indices->hv c) c-hv) :commutative?
        (= c c2) :retrieve-source
        [(similarity a
                     (indices->hv
                      (map #(mod % 500)
                           (f// (f/- c (hv->indices b)) n))))
         (= a
            (indices->hv (map #(mod % 500)
                              (f// (f/- c (hv->indices b)) n))))
         (= a
            (indices->hv (map #(mod % 500)
                              (f// (f/- (hv->indices (indices->hv
                                                      c))
                                        (hv->indices b))
                                   n))))
         (similarity a
                     (indices->hv (map #(mod % 500)
                                       (f// (f/- (hv->indices
                                                  (indices->hv
                                                   c))
                                                 (hv->indices b))
                                            n))))]
        (similarity b
                    (indices->hv
                     (map #(mod % 500)
                          (f/- c (f/* (hv->indices a) n)))))
        (= b
           (indices->hv (map #(mod % 500)
                             (f/- c
                                  (f/* (hv->indices a) n)))))])])

  ([1/3
    [0.0 0.0 1.0 :commutative? false :retrieve-source
     [1.0 true false 0.25] 1.0 true]]
   [1/2
    [0.0 0.0 1.0 :commutative? false :retrieve-source
     [1.0 true false 0.7] 1.0 true]])

  (defn directed-bind [a b]
    (let [n (/ 1 2)]
      (indices->hv
       (map #(mod % 500)
            (f/+ (f/* (hv->indices a) n) (hv->indices b))))))

  (defn directed-unbind-source
    [c b]
    (let [n (/ 1 2)]
      (indices->hv (map #(mod % 500)
                        (f// (f/- (hv->indices c)
                                  (hv->indices b)) n)))))

  (defn directed-unbind-destination
    [c a]
    (let [n (/ 1 2)]
      (indices->hv (map #(mod % 500)
                        (f/- c (f/* (hv->indices a) n))))))

  (for [n (range 500)]
    (let [a (->hv)
          b (->hv)
          c (directed-bind a b)]
      (similarity a (directed-unbind-source c b))))

  (defn rotate [a n]
    (indices->hv (dtt/rotate (hv->indices a) [n])))



  (bind a (rotate b 1))

  (= (rotate b 1) (unbind (bind a (rotate b 1)) a))
  (= b (rotate (unbind (bind a (rotate b 1)) a) -1))
  (= (rotate (bind a b) 1)
     (bind (rotate a 1) (rotate b 1)))

  (bind a (rotate b 1)))
