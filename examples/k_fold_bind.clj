(ns k-fold-bind
  (:require
    [tech.v3.datatype.functional :as f]
    [tech.v3.datatype :as dtype]
    [tech.v3.tensor :as dtt]
    [tech.v3.datatype.bitmap :as bitmap]
    [fastmath.random :as fm.rand]
    [fastmath.core :as fm]
    [bennischwerdtner.sdm.sdm :as sdm]
    [bennischwerdtner.hd.binary-sparse-segmented :as hd]
    [bennischwerdtner.pyutils :as pyutils]
    [tech.v3.datatype.unary-pred :as unary-pred]
    [tech.v3.datatype.argops :as dtype-argops]
    [bennischwerdtner.hd.codebook-item-memory :as codebook]
    [bennischwerdtner.hd.ui.audio :as audio]
    [bennischwerdtner.hd.data :as hdd]))

(alter-var-root
 #'hdd/*item-memory*
 (constantly (codebook/codebook-item-memory 1000)))

(defn randomize-drop
    [hv drop-chance]
    (hd/indices->hv*
     (for [idx (hd/hv->indices hv)]
       (when (zero? (fm.rand/flip drop-chance)) [idx]))))

(def sdm
  (sdm/k-fold-sdm {:address-count (long 1e6)
                   :address-density 0.000006
                   :k-delays 2
                   :word-length (long 1e4)}))




;; Basic principle:
;; C is representing the bind of A and B.
;; If C also goes back to A, then this is an attractor that will recover A,B,C given a little bit of any of them
;; (but the query needs to be in the order A-B-C)
;;

;;
;; A ------> C
;; A -> B
;;      B -> C
;; 0    1    2
;;
;; delay lines
;;

(defprotocol TrajectoryResonator
  (bind [this xs])
  (resonate [this query]))


(def resonator
  (reify
    TrajectoryResonator
    (bind [this xs]
      (sdm/reset sdm)
      ;; in biology, this would come from the
      ;; randomness of the wires leading to C. In a
      ;; Von Neuman computer, this can come randomly,
      ;; the moment somebody looks 'C'
      ;; - Can be a random seed
      ;; - We are liscenced to mix an HD bind, why
      ;; not
      ;;
      (let [xs (concat xs (hd/bind* xs))]
        (doseq [[addr content] (partition 2 1 xs)]
          (sdm/write sdm addr content 1))
        ;; and the loopy thing:
        ;; C -> A
        (doseq [[addr content] [(last xs) (first xs)]]
          (sdm/write sdm addr content 1))))
    (resonate [this query]
      (sdm/reset sdm)
      ;; [ A' , B' , C' ]
      (reductions
       (fn [query n]
         (into []
               (reductions
                (fn [q]
                  (:result (sdm/lookup sdm q 1)))
                (first query)
                query)))
       (range 3)))))

(do
  (sdm/reset sdm)
  (sdm/write sdm (hdd/clj->vsa* :a) (hdd/clj->vsa* :b) 1)
  (sdm/write sdm (hdd/clj->vsa* :b) (hdd/clj->vsa* [:* :a :b]) 1)
  ;; the loopy thing:
  (sdm/write
   sdm
   (hdd/clj->vsa* [:* :a :b])
   (hdd/clj->vsa* :a) 1))



;; resonate:

(def query [(hdd/clj->vsa* :a) (hdd/clj->vsa* :b) (hdd/clj->vsa* [:* :a :b])])

(def query [;; (hdd/clj->vsa* #{:a :b})
            (hd/->seed)
            (hdd/clj->vsa* #{:a :b})
            (hdd/clj->vsa* [:* :a :b])])

(do
  (sdm/reset sdm)
  (map report
       (take 3
             (iterate
              (fn [result-register]
                (pyutils/torch->jvm
                 (:result
                  (sdm/lookup sdm result-register 1 1))))
              (hdd/clj->vsa* :a)))))


(do
  (sdm/reset sdm)
  (sdm/write sdm (hdd/clj->vsa* :a) (hdd/clj->vsa* :b) 1)
  (sdm/write sdm (hdd/clj->vsa* :b) (hdd/clj->vsa* :c) 1)
  (sdm/write sdm (hdd/clj->vsa* :c) (hdd/clj->vsa* [:* :a :b :c]) 1)
  ;; the loopy thing:
  (sdm/write
   sdm
   (hdd/clj->vsa* (hdd/clj->vsa* [:* :a :b :c]))
   (hdd/clj->vsa* :a) 1))


(let [reader-frame 4
      query
      ;; [(hd/->seed) (hdd/clj->vsa* #{:a :b})
      ;; (hdd/clj->vsa* [:* :a :b])]
      [(hd/->seed)
       (hd/->seed)
       (hd/->seed)
       (hdd/clj->vsa* [:* :a :b :c])]]
  (do
    (sdm/reset sdm)
    (take reader-frame
          (drop
           (dec reader-frame)
           (map report
                (let [outcome
                      (reductions
                       (fn [result-register n]
                         (pyutils/torch->jvm
                          (:result
                           (sdm/lookup
                            sdm
                            (hd/superposition
                             result-register
                             ;; (hdd/clj->vsa*
                             ;; [:-- [:+ :a :b]
                             ;; 0.5])
                             (hdd/clj->vsa*
                              [:--
                               (nth query
                                    (mod n (count query)))
                               0.5]))
                            1
                            1))))
                       (first query)
                       (rest (range (inc (* 2
                                            reader-frame)))))]
                  (audio/listen! outcome)
                  outcome))))))










(do
  (sdm/reset sdm)
  (->> (reductions (fn [query n]
                     (into []
                           (reductions
                            (fn [result-register q]
                              (pyutils/torch->jvm
                               (:result
                                (sdm/lookup
                                 sdm
                                 (randomize-drop
                                  (hd/superposition
                                   result-register
                                   q)
                                  0.5)
                                 1
                                 1))))
                            (first query)
                            (next query))))
                   query
                   (range 10))
       last
       (map report)))














;; resonate:

(def query
  [(hd/->empty)
   (hdd/clj->vsa* :b)
   (hd/drop (hdd/clj->vsa* [:* :a :b]) 0.5)])


(def report
  (fn [x]
    (or (hdd/cleanup x)
        (when (< 0.95
                 (hd/similarity (hdd/clj->vsa* [:* :a :b])
                                x))
          :c)
        (when (< 0.95
                 (hd/similarity (hdd/clj->vsa* [:* :a :b
                                                :c])
                                x))
          :d))))

(let [query [(hd/->empty)
             (hd/drop (hdd/clj->vsa* :b) 0.8)
             (hd/drop (hdd/clj->vsa* [:* :a :b]) 0.8)]]
  (->> (do (sdm/reset sdm)
           ;; (sdm/lookup sdm (hd/->empty) 1 1)
           ;; (sdm/lookup sdm (hd/->empty) 1 1)
           ;; (sdm/lookup sdm (hdd/clj->vsa* [:* :a
           ;; :b])
           ;; 1 1)
           (reductions (fn [query n]
                         (into []
                               (reductions
                                (fn [result-register q]
                                  (pyutils/torch->jvm
                                   (:result
                                    (sdm/lookup
                                     sdm
                                     ;; (randomize-drop
                                     ;;  (hd/superposition
                                     ;;   result-register
                                     ;;   q)
                                     ;;  0.5)
                                     (hd/superposition
                                      result-register
                                      q)
                                     1
                                     1))))
                                (first query)
                                (next query))))
                       query
                       (range 3)))
       last
       (map report)))


(do (sdm/reset sdm)
    (let [xs (map hdd/clj->vsa* (range 3))]
      (let [xs (concat xs [(hd/bind* xs)])]
        (doseq [[addr content] (partition 2 1 xs)]
          (sdm/write sdm addr content 1))
        ;; and the loopy thing:
        ;; C -> A
        (sdm/write sdm (last xs) (first xs) 1))))

(let [c (hdd/clj->vsa* [:* 0 1 2])
      report (fn [x]
               (or (hdd/cleanup x)
                   (when (< 0.95 (hd/similarity c x)) :c)))
      query [(hdd/clj->vsa* 0)
             (hdd/clj->vsa* 1)
             (hdd/clj->vsa* 2)
             c]]
  (->> (do (sdm/reset sdm)
           (reductions (fn [query n]
                         (into []
                               (reductions
                                 (fn [result-register q]
                                   (pyutils/torch->jvm
                                     (:result
                                       (sdm/lookup
                                         sdm
                                         ;; (randomize-drop
                                         ;;  (hd/superposition
                                         ;;   result-register
                                         ;;   q)
                                         ;;  0.5)
                                         (hd/superposition
                                           result-register
                                           q)
                                         1
                                         1))))
                                 (first query)
                                 (next query))))
                       query
                       (range 15)))
       last
       (map report)))


(hdd/cleanup (pyutils/torch->jvm (:result (sdm/lookup sdm (hdd/clj->vsa* [:* 1 2 3]) 1 1))))



(let [reader-frame 3
      alphabet-n 15
      sdm (sdm/k-fold-sdm {:address-count (long 1e6)
                           :address-density 0.000006
                           :k-delays 2
                           :word-length (long 1e4)})
      factors (into []
                    (for [n (range 1)]
                      (into []
                            (repeatedly (dec reader-frame)
                                        #(rand-int
                                          alphabet-n)))))
      factor-bind
      (into
       []
       (for [f factors]
         (concat f [(hd/bind* (map hdd/clj->vsa* f))])))]
  (doseq [xs (map hdd/clj->vsa* factor-bind)]
    (sdm/reset sdm)
    (let [xs (concat xs [(first xs)])]
      (doseq [[addr content] (partition 2 1 xs)]
        (sdm/write sdm addr content 1))))
  (let [[x y c] (rand-nth factor-bind)]
    (hd/similarity
     (hdd/clj->vsa* y)
     (pyutils/torch->jvm
      (:result
       (sdm/lookup sdm (hdd/clj->vsa* [:* x y]) 1 1))))))








(let [reader-frame 3
      alphabet-n 15
      sdm (sdm/k-fold-sdm {:address-count (long 1e6)
                           :address-density 0.000006
                           :k-delays 2
                           :word-length (long 1e4)})
      factors (into []
                    (for [n (range 1)]
                      (into []
                            (repeatedly (dec reader-frame)
                                        #(rand-int
                                          alphabet-n)))))
      factor-bind
      (into
       []
       (for [f factors]
         (concat f [(hd/bind* (map hdd/clj->vsa* f))])))]
  (doseq [xs (map hdd/clj->vsa* factor-bind)]
    (sdm/reset sdm)
    (let [xs (concat xs [(first xs)])]
      (doseq [[addr content] (partition 2 1 xs)]
        (sdm/write sdm addr content 1))))
  (let [factors-and-bind (rand-nth factor-bind)
        query (repeatedly reader-frame hd/->seed)
        query (concat (drop-last 1 query)
                      [(last factors-and-bind)])]
    (do
      (sdm/reset sdm)
      [ ;; query
       factors-and-bind
       (take-last
        reader-frame
        (map (fn [x]
               (when x
                 (:k (first (hdd/cleanup-verbose x 0.9)))))
             (let [outcome
                   (reductions
                    (fn [result-register n]
                      (pyutils/torch->jvm
                       (:result
                        (sdm/lookup
                         sdm
                         (hd/superposition
                          result-register
                          ;; (hdd/clj->vsa*
                          ;; [:-- [:+ :a :b]
                          ;; 0.5])
                          (hdd/clj->vsa*
                           [:--
                            (nth query
                                 (mod n
                                      (count query)))
                            0.5]))
                         1
                         1))))
                    (first query)
                    (rest (range (inc (* 2
                                         reader-frame)))))]
               (audio/listen! outcome)
               outcome)))])))


;; -------------------------------------

(let [reader-frame 5
      alphabet-n 15
      sdm (sdm/k-fold-sdm {:address-count (long 1e6)
                           :address-density 0.000006
                           :k-delays 2
                           :word-length (long 1e4)})
      factors (into []
                    (for [n (range 1)]
                      (into []
                            (repeatedly (dec reader-frame)
                                        #(rand-int
                                          alphabet-n)))))
      factor-bind
      (into
       []
       (for [f factors]
         (concat f [(hd/bind* (map hdd/clj->vsa* f))])))]
  (doseq [xs (map hdd/clj->vsa* factor-bind)]
    (sdm/reset sdm)
    (let [xs (concat xs [(first xs)])]
      (doseq [[addr content] (partition 2 1 xs)]
        (sdm/write sdm addr content 1))))
  (let [factors-and-bind (rand-nth factor-bind)
        query (repeatedly reader-frame hd/->seed)
        query (concat (drop-last 1 query)
                      [(last factors-and-bind)])]
    (do
      (sdm/reset sdm)
      [ ;; query
       factors-and-bind
       (take-last
        reader-frame
        (map (fn [x]
               (when x
                 (:k (first (hdd/cleanup-verbose x 0.9)))))
             (let [outcome
                   (reductions
                    (fn [result-register n]
                      (pyutils/torch->jvm
                       (:result
                        (sdm/lookup
                         sdm
                         (hd/superposition
                          result-register
                          ;; (hdd/clj->vsa*
                          ;; [:-- [:+ :a :b]
                          ;; 0.5])
                          (hdd/clj->vsa*
                           [:--
                            (nth query
                                 (mod n
                                      (count query)))
                            0.5]))
                         1
                         1))))
                    (first query)
                    (rest (range (inc (* 3 reader-frame)))))]
               (audio/listen! outcome)
               outcome)))])))


(let [reader-frame 4
      alphabet-n 15
      sdm (sdm/k-fold-sdm {:address-count (long 1e6)
                           :address-density 0.000006
                           :k-delays 5
                           :word-length (long 1e4)})
      factors (into []
                    (for [n (range 10)]
                      (into []
                            (repeatedly (dec reader-frame)
                                        #(rand-int
                                          alphabet-n)))))
      factor-bind
      (into
       []
       (for [f factors]
         (concat f [(hd/bind* (map hdd/clj->vsa* f))])))]
  (doseq [xs (map hdd/clj->vsa* factor-bind)]
    (sdm/reset sdm)
    (let [xs (concat xs [(first xs)])]
      (doseq [[addr content] (partition 2 1 xs)]
        (sdm/write sdm addr content 1))))
  (let [factors-and-bind (rand-nth factor-bind)
        query (repeatedly reader-frame hd/->seed)
        query (concat (drop-last 1 query)
                      [(last factors-and-bind)])]
    (do
      (sdm/reset sdm)
      [ ;; query
       factors-and-bind
       (let [out
             (into
              []
              (take-last
               reader-frame
               (map (fn [x]
                      (when x
                        (:k (first (hdd/cleanup-verbose
                                    x
                                    0.9)))))
                    (let [outcome
                          (reductions
                           (fn [result-register n]
                             (pyutils/torch->jvm
                              (:result
                               (sdm/lookup
                                sdm
                                (hd/superposition
                                 result-register
                                 ;; (hdd/clj->vsa*
                                 ;; [:-- [:+ :a
                                 ;; :b]
                                 ;; 0.5])
                                 (hdd/clj->vsa*
                                  [:-- (nth query (mod n (count query))) 0.8]))
                                1
                                1))))
                           (first query)
                           (rest
                            (range
                             (inc
                              (* 3 reader-frame)))))]
                      (audio/listen! outcome)
                      outcome))))
             first-nil-idx
             (first (keep identity
                          (map-indexed (fn [idx e]
                                         (when-not e idx))
                                       out)))
             outcomeseq (take (dec reader-frame)
                              (drop (inc first-nil-idx)
                                    (cycle out)))]
         [outcomeseq
          (= (into [] outcomeseq)
             (into [] (butlast factors-and-bind)))])])))




(let [reader-frame 5
      alphabet-n 15
      sdm (sdm/k-fold-sdm {:address-count (long 1e6)
                           :address-density 0.000007
                           :k-delays 5
                           :word-length (long 1e4)})
      factors (into []
                    (for [n (range 1)]
                      (into []
                            (repeatedly (dec reader-frame)
                                        #(rand-int
                                          alphabet-n)))))
      factor-bind
      (into
       []
       (for [f factors]
         (concat f [(hd/bind* (map hdd/clj->vsa* f))])))]
  (doseq [xs (map hdd/clj->vsa* factor-bind)]
    (sdm/reset sdm)
    (let [xs (concat xs [(first xs)])]
      (doseq [[addr content] (partition 2 1 xs)]
        (sdm/write sdm addr content 1))))
  (let [factors-and-bind (rand-nth factor-bind)
        query (repeatedly reader-frame hd/->empty)
        query (concat (drop-last 1 query)
                      [(last factors-and-bind)])
        query
        (let [idx (rand-int (dec (count query)))]
          (assoc
           query
           (hd/drop (nth factors-and-bind idx) 0.8)))]
    (do
      (sdm/reset sdm)
      [ ;; query
       factors-and-bind
       (let [out
             (into
              []
              (take-last
               reader-frame
               (map (fn [x]
                      (when x
                        (:k (first (hdd/cleanup-verbose
                                    x
                                    0.9)))))
                    (let [outcome
                          (reductions
                           (fn [result-register n]
                             (pyutils/torch->jvm
                              (:result
                               (sdm/lookup
                                sdm
                                (hd/superposition
                                 result-register
                                 ;; (hdd/clj->vsa*
                                 ;; [:-- [:+ :a
                                 ;; :b]
                                 ;; 0.5])
                                 (hdd/clj->vsa*
                                  [:--
                                   (nth
                                    query
                                    (mod
                                     n
                                     (count
                                      query)))
                                   0.5]))
                                1
                                1))))
                           (first query)
                           (rest
                            (range
                             (inc
                              (* 3 reader-frame)))))]
                      ;; (audio/listen! outcome)
                      outcome))))
             first-nil-idx
             (first (keep identity
                          (map-indexed (fn [idx e]
                                         (when-not e idx))
                                       out)))
             outcomeseq (take (dec reader-frame)
                              (drop (inc first-nil-idx)
                                    (cycle out)))]
         [outcomeseq
          (= (into [] outcomeseq)
             (into [] (butlast factors-and-bind)))])])))


































;; In the next version, the k-delays are controlled with a clock-wire
;; (like with a synfire chain arrangement)
;; This allows the user to control time, (time actually was the 'trajectory')
;;
;; - It's cool that all elements not served by either a clock wire or address input
;;   are allowed to stay, it implements a cache
;;
