;; -----------------------
;; WIP
;; ----------------------

(ns sparse-resonator
  (:require
   [tech.v3.datatype.functional :as f]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [tech.v3.datatype.bitmap :as bitmap]
   [fastmath.random :as fm.rand]
   [fastmath.core :as fm]
   [bennischwerdtner.sdm.sdm :as sdm]
   [bennischwerdtner.hd.core :as hd]
   [bennischwerdtner.pyutils :as pyutils]
   [tech.v3.datatype.unary-pred :as unary-pred]
   [tech.v3.datatype.argops :as dtype-argops]
   [bennischwerdtner.hd.codebook-item-memory :as codebook]
   [bennischwerdtner.hd.ui.audio :as audio]
   [bennischwerdtner.hd.data-next :as hdd]
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python :refer [py. py..] :as py]))

(do (require-python '[numpy :as np])
    (require-python '[torch :as torch])
    (require-python '[torch.sparse :as torch.sparse]))


;;
;; Problem:
;; x is the result of binding 3 vecs from 3 codebooks
;;
;;
;; Find whatever the contributing factors are from
;; the codebooks
;;
;;

;; D is the hdv dimension = codebook dimension
;; k is the non zero bits in hdv
;; M is the alphabet length (= codebook size)
;; depth is the count of factors (= codebook count)

(def depth 3)
(def alphabet-length 10)
(def alphabet (range alphabet-length))
;; a
;; chose depending on D, k
(def attention-sparsification 4)

(defn ->codebook [size]
  (py.. (hd/seed size) (to :dtype torch/float16)))

(defn codebook-weight [codebook x]
  (torch/mv codebook x))

(defn randonmess [shape]
  (torch/le (torch/randn shape :device pyutils/*torch-device*) 0))

(defn l-infinity-similarity
  [a b]

  ;; (- 1
  ;;    (py..
  ;;        (torch/abs (torch/sub a b))
  ;;        (view 20 500)))
  )

;; (l-infinity-similarity)






;; A, B, C
;; A = { a0, ... aM-1 }
;; ..

(def codebooks
  (into []
        (for [n (range depth)]
          (->codebook alphabet-length))))

(defn initialize
  [codebooks]
  {:estimates (into []
                    (map
                     ;; superimpose all entries in
                     ;; code book an thin
                     (fn [codebook]
                       (py.. (hd/thin (hd/superposition
                                       codebook))
                         (to :dtype torch/float16)))
                     codebooks))})

(defn sparsify-weights
  [w]
  (let [topk-output (torch/topk w attention-sparsification)]
    ;; [ 0 1 2 0 ]
    (torch/scatter (torch/zeros_like w)
                   0
                   (py.. topk-output -indices)
                   (py.. topk-output -values))))


(defn weighted-bundle
  [weights codebook]
  (let [w (torch/mul codebook (torch/unsqueeze weights 1))]
    (hd/thin (hd/drop (hd/superposition w) 0.2))))


;; roughly:
;; The similarity for each vec in codebook is 1/M (M is the number items in codebook)
;;

(def estimates (:estimates (initialize codebooks)))

(comment
  (codebook-weight (first codebooks) (py/get-item (first codebooks) 0))
  ;; tensor([20.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], device='cuda:0',
  ;;      dtype=torch.float16)

  (py..
      (codebook-weight (first codebooks)
                       (first estimates))
      (size 0))

  ;; tensor([0., 5., 3., 4., 3., 2., 1., 2., 2., 1.], device='cuda:0',
  ;;        dtype=torch.float16)

  ;; (/ 20 10)

  (def weights (codebook-weight (first codebooks) (first estimates)))

  (def attention-weights
    (->
     (torch/add weights (randonmess (py.. weights -shape)))
     (sparsify-weights)))

  ;;
  (* 6 20)
  120

  (torch/sum
   (py/get-item
    (torch/mul
     (first codebooks)
     ;; tensor([[0., 0., 0.,  ..., 0., 0., 0.],  ; <- 0
     ;;      [0., 0., 0.,  ..., 0., 0., 0.],     ; <- 6
     ;;      [0., 0., 0.,  ..., 0., 0., 0.],
     ;;      ...,
     ;;      [0., 0., 0.,  ..., 0., 0., 0.],
     ;;      [0., 0., 0.,  ..., 0., 0., 0.],
     ;;      [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', dtype=torch.float16)

     ;; tensor([0., 6., 4., 5., 4., 3., 0., 0., 0., 0.], device='cuda:0',

     (torch/unsqueeze attention-weights 1))
    1))
  ;; tensor(120., device='cuda:0', dtype=torch.float16)


  (apply +
         (let [weighted-codebook
               (torch/mul
                (first codebooks)
                (torch/unsqueeze attention-weights 1))]
           ;; (torch/sum (hd/superposition weighted-codebook))

           ;; (hd/similarity
           ;;  (second (first codebooks))
           ;;  (hd/thin
           ;;   (hd/superposition weighted-codebook)))

           ;; (torch/sum
           ;;  (torch/clamp_max (hd/superposition weighted-codebook) 1))
           (def weighted-codebook weighted-codebook)

           ;; (for [hdv (first codebooks)]
           ;;   (hd/similarity
           ;;    ;; (py/get-item (first codebooks) 1)
           ;;    ;; (py/get-item (first codebooks) 3)
           ;;    hdv
           ;;    (hd/thin
           ;;     (torch/sum
           ;;      (torch/stack
           ;;       [(hd/superposition weighted-codebook)
           ;;        (randonmess [(long 1e4)])
           ;;        (randonmess [(long 1e4)])
           ;;        (randonmess [(long 1e4)])
           ;;        (randonmess [(long 1e4)])])
           ;;      :dim 0))))
           (for [hdv (first codebooks)]
             (hd/similarity
              ;; (py/get-item (first codebooks) 1)
              ;; (py/get-item (first codebooks) 3)
              hdv
              (hd/thin
               (hd/drop
                (hd/superposition weighted-codebook)
                0.6))))
           ;; (for [hdv (first codebooks)]
           ;;   (hd/similarity
           ;;    ;; (py/get-item (first codebooks) 1)
           ;;    ;; (py/get-item (first codebooks) 3)
           ;;    hdv
           ;;    (hd/thin
           ;;     (hd/superposition weighted-codebook))))
           ))

  '(0 11/20 1/10 3/10 1/10 0 0 0 1/20 0)







  ;; 1/5
  ;; (- 1 (float (/ 4 5)))












  ;; -------------------------
  ;; sparsify:
  (let [w (torch/tensor [0 1 2 0])
        sparsification-factor 2
        topk-output (torch/topk w sparsification-factor)]
    ;; [ 0 1 2 0 ]
    (torch/scatter (torch/zeros_like w)
                   0
                   (py.. topk-output -indices)
                   (py.. topk-output -values))))





  ;; torch.return_types.topk(
  ;;                         values=tensor([2, 1]),
  ;;                         indices=tensor([2, 1]))













;;








(defn iterative-factorization
  [query-x]
  (fn [{:keys [estimates]} round]
    {:estimates
       ;;
       ;; new estimate
       (into
         []
         (for [i (range (count estimates))
               :let [other-idxs (disj (into #{}
                                            (range
                                              (count
                                                estimates)))
                                      i)]]
           (let [new-estimate (apply hd/unbind
                                query-x
                                (map estimates other-idxs))
                 new-estimate (py.. new-estimate
                                    (to :dtype
                                        torch/float16))
                 ;; codebook attention w'
                 weights (codebook-weight (codebooks i)
                                          new-estimate)
                 ;; weights
                 weights (torch/add weights
                                    (randonmess (py..
                                                  weights
                                                  -shape)))
                 ;; (torch/add
                 ;;  weights
                 ;;  (torch/le
                 ;;   (torch/rand_like
                 ;;    weights
                 ;;    :device
                 ;;    pyutils/*torch-device*)
                 ;;   0.8))
                 ;;
                 ;; sparsify
                 weights (sparsify-weights weights)
                 new-estimate
                   (weighted-bundle weights (codebooks i))]
             (py.. new-estimate
                   (to :dtype torch/float16)))))}))




(comment

  (py.. (apply hd/bind (map first codebooks)) -dtype)


  (defn report
    [{:keys [estimates]}]
    {:estimates estimates
     :outcomes (into []
                     (map (fn [estimate codebook]
                            (py.. (torch/argmax
                                   (codebook-weight
                                    codebook
                                    estimate))
                              (item)))
                          estimates
                          codebooks))
     :similarities (into []
                         (map (fn [estimate input]
                                (hd/similarity estimate
                                               input))
                              estimates
                              (map first codebooks)))})


  (f/mean
   (map
    #(= [0 0 0] %)
    (for [n (range 25)]
      (:outcomes
       (last
        (map report
             (take
              15
              (let
                  [x (apply hd/bind (map first codebooks))]
                  (reductions
                   (iterative-factorization x)
                   (initialize codebooks)
                   (range))))))))))





  ;; 0.8
  ;; 0.6
  ;; 0.8666666666666667
  ;; 0.78






  )


(comment

  (do (def alphabet-length 10)
      (def codebooks
        (into []
              (for [n (range depth)]
                (->codebook alphabet-length))))
      (def attention-sparsification (/ alphabet-length 10))
      (defn weighted-bundle
        [weights codebook]
        (let [w (torch/mul codebook
                           (torch/unsqueeze weights 1))]
          ;; (hd/thin (hd/superposition w))
          (hd/thin (hd/drop (hd/superposition w) 0.5))))
      (defn initialize
        [codebooks]
        {:estimates (into []
                          (map
                           ;; superimpose all entries in
                           ;; code book an thin
                           (fn [codebook]
                             (py..
                                 (hd/thin (hd/superposition codebook))
                               ;; 0.9
                                 (to :dtype
                                     torch/float16)))
                           codebooks))})
      (f/mean
       (map

        (for [n (range 10)]
          (:outcomes
           (last (map report
                      (take 15
                            (let [x (apply hd/bind (map first codebooks))]
                              (reductions
                               (iterative-factorization x)
                               (initialize codebooks)
                               (range)))))))))))


  )






(comment
  (apply hd/unbind (hd/seed) [(hd/seed) (hd/seed)])
  (def a (hd/seed))
  (def b (hd/seed))
  (def c (hd/seed))
  (def x (hd/bind a b c))
  (hd/similarity c (hd/unbind x a b)))



(comment
  (for [hdv
        (first codebooks)]
    (hd/similarity
     (first (:estimates (initialize codebooks)))
     hdv))
  '(0 1/4 3/20 1/5 3/20 1/10 1/20 1/10 1/10 1/20))

(comment
  (def cb (hd/seed 2))
  (hd/similarity (hd/superposition cb) (py/get-item cb 0))
  (hd/similarity (hd/superposition cb) (hd/seed))
  (hd/similarity (hd/superposition cb) (py/get-item cb 1))
  (hd/similarity (hd/thin (hd/superposition cb)) (py/get-item cb 1)))






(hdd/clj->vsa* [:> :a 1])

(codebook/codebook-cleanup-verbose
 (first codebooks)
 (hdd/clj->vsa* :left)
 0)

(comment
  (time
   (do
     (def m (sdm/->sdm
             {:address-count (long 1e6)
              :address-density 0.000003
              :word-length (long 1e4)}))
     (dotimes [n 500]
       (sdm/write m (hd/seed) (hd/seed) 1))
     (dotimes [n 500]
       (sdm/lookup m (hd/seed) 1 1))))
  )
