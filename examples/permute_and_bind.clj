(require '[bennischwerdtner.hd.binary-sparse-segmented :as hd])
(require '[tech.v3.tensor :as dtt])
(require '[tech.v3.datatype.functional :as f])

(let [m (atom {})]
    (defn clj->vsa
      [obj]
      (or (@m obj) ((swap! m assoc obj (hd/->seed)) obj)))
    (defn cleanup
      [q]
      (:k (first (filter (comp #(< 0.1 %) :similarity)
                         (sort-by :similarity
                                  #(compare %2 %1)
                                  (into []
                                        (pmap
                                         (fn [[k v]]
                                           {:k k
                                            :similarity
                                            (hd/similarity v q)
                                            :v v})
                                         @m))))))))


;; Observe, with a binary {-1,1} encoding and multiply-add VSA, permutation distributes over binding
;;

(let
    [a (dtt/->tensor [-1 -1 1 1])
     b (dtt/->tensor [1 -1 -1 1])
     c (dtt/->tensor [1 1 -1 -1])
     bind (fn [x y] (f/* x y))
     permute (fn [x] (dtt/rotate x [1]))]
  (= (bind (permute a) (permute b)) (permute (bind a b))))
true

(dotimes
    [_ 100]
    (assert (let [->hv (fn []
                         (dtt/->tensor
                          (dtt/compute-tensor
                           [(long 1e4)]
                           (fn [_]
                             (rand-nth [-1 1])))))
                  a (->hv)
                  b (->hv)
                  c (->hv)
                  bind (fn [x y] (f/* x y))
                  permute (fn [x] (dtt/rotate x [1]))]
              (= (bind (permute a) (permute b))
                 (permute (bind a b))))))
nil


;; This is not the case here.  👈
;; ---------------------------------------------------


;; Here, imagine that each permute is contributing a 'permute' offset.
;; In the above, roughly, there are 2 permutes, so the result is permuted twice.

(do
  (assert
   (=
    ;; This concept becomes clear, when we
    ;; compose a bind with 2 times the permute
    ;; unit vector
    (hd/bind* [(clj->vsa :a) (clj->vsa :b)
               (hd/unit-vector-n 1)
               (hd/unit-vector-n 1)])
    ;; ------------
    ;;  These are all equivalent
    ;; ------------
    (hd/bind (hd/permute (clj->vsa :a))
             (hd/permute (clj->vsa :b)))
    (hd/permute (hd/bind (hd/permute (clj->vsa :a))
                         (clj->vsa :b)))
    (hd/permute (hd/permute (hd/bind (clj->vsa :a)
                                     (clj->vsa :b))))))
  ;; bind and permute are still commutative and
  ;; associative
  ;; (clear because permute is equivalent to a bind)
  ;; Bind is obviously associative and commutative, see
  ;; [[hd/bind*]].
  (assert
   (= (hd/bind* [(clj->vsa :a) (clj->vsa :b)
                 (hd/unit-vector-n 1)])
      (hd/bind (clj->vsa :a)
               (hd/bind (clj->vsa :b)
                        (hd/unit-vector-n 1)))
      (hd/bind (clj->vsa :a) (hd/permute (clj->vsa :b)))
      (hd/permute (hd/bind (clj->vsa :a) (clj->vsa :b))))))



;; ------------------------
;; This is similar to the need for a non commutative bind, which we have available, see
;; data.clj -> ->directed-edge
;;
;;
