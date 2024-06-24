
(require '[bennischwerdtner.hd.binary-sparse-segmented :as hd])
(require '[tech.v3.datatype.functional :as f])

;;
;; A - Ambiguity primitives
;;
;;
;;

(defn mix
  ([a b & args] (hd/thin (apply f/+ a b args)))
  ([a b] (hd/thin (hd/bundle a b))))

(def possibly mix)

(comment
  (hd/similarity (mix (->prototype :a)
                      (->prototype :b)
                      (->prototype :c)
                      (->prototype :d))
                 (->prototype :d))
  0.29)

(def neither (fn [a b] (hd/bind a b)))

(def roughly
  (fn [a amount-of-a] (hd/weaken a (- 1 amount-of-a))))

(defn mostly
  ([a b] (mostly a b 0.3))
  ([a b amount-of-b]
   (hd/thin (hd/bundle a (roughly b amount-of-b)))))

(defn never [e b]
  (hd/thin (f/- e b)))

(def impossibly never)

(defn non-sense [] (hd/->hv))

;; I think there is something deep about the concept that
;; non-sense and gensym are the same operation
(def create non-sense)

(comment
  (let [a (hd/->hv)
        b (hd/->hv)]
    ;; then it is 50:50
    [(hd/similarity a (mostly a b 1.0))
     (hd/similarity b (mostly a b 1.0))])

  (let [a (hd/->hv)
        b (hd/->hv)]
    ;; this is mostly a
    [(hd/similarity a (mostly a b 0.5))
     (hd/similarity b (mostly a b 0.5))])
  [0.79 0.21]

  ;; I guess 0.3 is at the limit of still being similar to b
  (let [a (hd/->hv)
        b (hd/->hv)]
    ;; this is mostly a
    [(hd/similarity a (mostly a b 0.3))
     (hd/similarity b (mostly a b 0.3))])
  [0.77 0.19])



(require '[bennischwerdtner.hd.binary-sparse-segmented :as hd])


(let [a (hd/->seed)
      b (hd/->seed)
      c (hd/->seed)]
  (hd/similarity
   (hd/bind (hd/thin (hd/bundle a b)) c)
   (hd/thin (hd/bundle (hd/bind a c) (hd/bind b c)))))



(require '[bennischwerdtner.hd.binary-sparse-segmented :as hd])

(let [a (hd/->seed)
      b (hd/->seed)
      c (hd/->seed)]
  (=
   (hd/bind c (hd/bundle a b))
   (hd/bundle (hd/bind a c) (hd/bind b c))))
true

(let [a (hd/->seed)
      b (hd/->seed)
      c (hd/->seed)]
  (=
   (hd/thin (hd/bind c (hd/bundle a b)))
   (hd/thin (hd/bundle (hd/bind a c) (hd/bind b c)))))
