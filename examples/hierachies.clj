(ns hierachies
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


;;
;; Sequence hierachies:
;; ---------------


;; Idea I:
;;
;; Make the first element 'stand for' the nested sequence
;;

;;
;; terminals:
;;
;;      [ a b c d e f g ]
;;      [ h j k l ...   ]
;;
;;
;; higher order sequence:
;;
;;
;;  [ a , h , ... ]


;; Questions:
;;
;; - how do you differentiate between pointers to a seq and the terminal?
;; -
;;


(def sdm (sdm/k-fold-sdm {:address-count (long 1e6)
                          :address-density 0.000006
                          :k-delays 5
                          :word-length (long 1e4)}))

(def alphabet (into [] (map keyword (map str (map char (range 97 123))))))

(partition-all 7 alphabet)
'((:a :b :c :d :e :f :g)
 (:h :i :j :k :l :m :n)
 (:o :p :q :r :s :t :u)
 (:v :w :x :y :z))

(do
  (sdm/reset sdm)
  (for [subseq (partition-all 7 (map hdd/clj->vsa alphabet))]
    (doseq [[addr content] (partition 2 1 subseq)]
      (sdm/write sdm addr content 1))))

(do
  (sdm/reset sdm)
  (let [higher-order-seq
          (map first
            (partition-all 7 (map hdd/clj->vsa alphabet)))]
    (doseq [[addr content] (partition 2 1 higher-order-seq)]
      (sdm/write sdm addr content 1))))

(do
  (sdm/reset sdm)
  ;; querying with top-k 2 results in both the
  ;; terminal and the higher order sequence
  (let [a-kont (:result (sdm/lookup sdm (hdd/clj->vsa* :a) 2 1))]
    (->>
     (sdm/converged-lookup-impl
      sdm
      a-kont
      {:decoder-threshold 1
       :max-steps (count alphabet)
       :stop? (fn [acc next-outcome]
                (when (< (:confidence next-outcome) 0.05)
                  {:stop-reason :low-confidence :success? false}))
       :top-k 1})
     :result-xs
     ;; (map :result)
     ;; (map pyutils/ensure-jvm)
     )))

(do
  (sdm/reset sdm)
  ;; querying with top-k 2 results in both the
  ;; terminal and the higher order sequence
  (let [a-kont (:result (sdm/lookup sdm (hdd/clj->vsa* :a) 2 1))]
    (map hdd/cleanup*
         (->>
          (sdm/converged-lookup-impl
           sdm
           a-kont
           {:decoder-threshold 1
            :max-steps (count alphabet)
            :stop? (fn [acc next-outcome]
                     (when (< (:confidence next-outcome) 0.5)
                       {:stop-reason :low-confidence :success? false}))
            :top-k 1})
          :result-xs
          (map :result)
          (map pyutils/ensure-jvm)))))

;; cool properties out of the box:
;; (note that it stopped after z when confidence was low)
;;

'(
  ;; superposition of terminal and higher order sequence
  (:h :b)
  (:c)
  (:d)
  (:e)
  (:f)
  (:g)
  (:i)
  (:j)
  (:k)
  (:l)
  (:m)
  (:n)
  (:p)
  (:q)
  (:r)
  (:s)
  (:t)
  (:u)
  (:w)
  (:x)
  (:y)
  (:z))




;;
;; let's say you *somehow* know that you can 'skip' `:b` ...
;; say we know that 'b' is a terminal, but we look for the higher order seq.
;;

(do
  (sdm/reset sdm)
  ;; querying with top-k 2 results in both the
  ;; terminal and the higher order sequence
  (let [a-kont (pyutils/ensure-jvm (:result (sdm/lookup sdm (hdd/clj->vsa* :a) 2 1)))
        ;; expressing the notion that you don't want to see `:b`
        a-kont (hdd/difference a-kont (hdd/clj->vsa* :b))]
    (map hdd/cleanup*
         (->>
          (sdm/converged-lookup-impl
           sdm
           a-kont
           {:decoder-threshold 1
            :max-steps (count alphabet)
            :stop? (fn [acc next-outcome]
                     (when (< (:confidence next-outcome) 0.5)
                       {:stop-reason :low-confidence :success? false}))
            :top-k 1})
          :result-xs
          (map :result)
          (map pyutils/ensure-jvm)))))

'((:h) (:o) (:v))


;; ----------------------------------
;; Idea II:
;;
;; - like idea I, but permute the higher order sequence for each hierachy
;; - biologically: this might represent the hyper vector projected in a different
;;   neuronal area.
;; - (but more complicated things with timings etc might go on).
;; - either way, perhaps the humble permute already gives us certain expressivity.
;;
;; - Downside: If you have multiple hierachical seqs, you need to know the exact level
;;   (permute correctly)
;; - I see this might in effect become a factorization problem, which might be suited to solve
;;   with a resonator-like net.
;;


(do
  (def sdm
    (sdm/k-fold-sdm {:address-count (long 1e6)
                     :address-density 0.000006
                     :k-delays 5
                     :word-length (long 1e4)}))
  (def alphabet
    (into []
          (map keyword
               (map str (map char (range 97 123))))))
  (sdm/reset sdm)
  ;;
  ;; terminal seq:
  ;;
  (doseq
      [subseq
       (partition-all 7 (map hdd/clj->vsa alphabet))]
      (doseq [[addr content] (partition 2 1 subseq)]
        (sdm/write sdm addr content 1)))

  ;;
  ;; higher order seq:
  ;;
  (do
    (sdm/reset sdm)
    (let [higher-order-seq
          (map first
               (partition-all 7 (map hdd/clj->vsa alphabet)))]
      ;; permute 1
      (doseq [[addr content]
              (partition 2 1 (map hd/permute higher-order-seq))]
        (sdm/write sdm addr content 1)))))



;; -------------

(do (sdm/reset sdm)
    (let [a-kont (:result
                   (sdm/lookup sdm (hdd/clj->vsa* :a) 1 1))]
      (map hdd/cleanup*
        (->> (sdm/converged-lookup-impl
               sdm
               a-kont
               {:decoder-threshold 1
                :max-steps (count alphabet)
                :stop? (fn [acc next-outcome]
                         (when (< (:confidence next-outcome)
                                  0.5)
                           {:stop-reason :low-confidence
                            :success? false}))
                :top-k 1})
             :result-xs
             (map :result)
             (map pyutils/ensure-jvm)))))

;; can query for terminal(s):

'((:b)
 (:c)
 (:d)
 (:e)
 (:f)
 (:g)
 (:i)
 (:j)
 (:k)
 (:l)
 (:m)
 (:n)
 (:p)
 (:q)
 (:r)
 (:s)
 (:t)
 (:u)
 (:w)
 (:x)
 (:y)
 (:z))


;; ------------
;; Now higher order seq usage:


(do (sdm/reset sdm)
    (let [ ;; query with p(a)
          a-kont
          (:result
           (sdm/lookup sdm (hdd/clj->vsa* [:> :a]) 1 1))]
      (map hdd/cleanup*
           ;; to cleanup you need to permute-inverse
           (map hd/permute-inverse
                (->> (sdm/converged-lookup-impl
                      sdm
                      a-kont
                      {:decoder-threshold 1
                       :max-steps (count alphabet)
                       :stop? (fn [acc next-outcome]
                                (when (< (:confidence
                                          next-outcome)
                                         0.5)
                                  {:stop-reason :low-confidence
                                   :success? false}))
                       :top-k 1})
                     :result-xs
                     (map :result)
                     (map pyutils/ensure-jvm))))))
'((:h) (:o) (:v))

;; Say you are capabable of knowing:
;; - I want a higher order sequence that starts with `:a`
;; - I want to skip 2 terminal sequences and list the 3rd.
;;
;; (skip a and h, find the seq starting with o...)
;;


(do
  (sdm/reset sdm)
  (let [ ;; query with p(a)
        a-kont
        (:result
         (sdm/lookup sdm (hdd/clj->vsa* [:> :a]) 1 1))
        higher-seq (->> (sdm/converged-lookup-impl
                         sdm
                         a-kont
                         {:decoder-threshold 1
                          :max-steps (count alphabet)
                          :stop? (fn [acc next-outcome]
                                   (when (< (:confidence
                                             next-outcome)
                                            0.5)
                                     {:stop-reason
                                      :low-confidence
                                      :success? false}))
                          :top-k 1})
                        :result-xs
                        (map :result)
                        (map pyutils/ensure-jvm))
        ;; in order to query 'one down' at this point,
        ;; you permute inverse once.
        ;; (could happen automagically in brain areas?)
        ;; -----
        ;; being able to do such a 'wait' / 'skip' kinda thing might be implemented with
        ;; higher order processes giving a rhythm, if the query elements flow in time
        ;; (which is the idea of k-fold sdm to begin with)
        ;; -----
        ;; Such a thing would be fundamentally useful, it would allow to percieve a sequence,
        ;; and 'jump ahead' in leaps of ~7.
        ;;
        terminal-query (hd/permute-inverse
                        (first (drop 1 higher-seq)))]
    ;; I needed to reset this, too.
    ;; but perhaps you actually have multiple per hierachical seq anyway?
    ;; (since permuting puts it into an unrelated domain anyway).
    ;; (then, you have an sdm for each permutation level and we are halfway to a resonator net)
    ;;
    (sdm/reset sdm)
    (map hdd/cleanup*
         (->> (sdm/converged-lookup-impl
               sdm
               terminal-query
               {:decoder-threshold 1
                :max-steps (count alphabet)
                :stop? (fn [acc next-outcome]
                         (when (< (:confidence next-outcome)
                                  0.5)
                           {:stop-reason :low-confidence
                            :success? false}))
                :top-k 1})
              :result-xs
              (map :result)
              (map pyutils/ensure-jvm)))))

'((:o) (:p) (:q) (:r) (:s) (:t) (:u) (:w) (:x) (:y) (:z))

;; I get of course the whole terminal seq,
;; But in brain, everything presumably chunks to 7 elements (7 gamma per theta cycle)
;;

(do
  (sdm/reset sdm)
  (let [a-kont
          (:result
            (sdm/lookup sdm (hdd/clj->vsa* [:> :a]) 1 1))
        higher-seq (->> (sdm/converged-lookup-impl
                          sdm
                          a-kont
                          {:decoder-threshold 1
                           :max-steps (count alphabet)
                           :stop? (fn [acc next-outcome]
                                    (when (< (:confidence
                                               next-outcome)
                                             0.5)
                                      {:stop-reason
                                         :low-confidence
                                       :success? false}))
                           :top-k 1})
                        :result-xs
                        (map :result)
                        (map pyutils/ensure-jvm))
        terminal-query (hd/permute-inverse
                         (first (drop 1 higher-seq)))]
    (sdm/reset sdm)
    (map hdd/cleanup*
      (->> (sdm/converged-lookup-impl
             sdm
             terminal-query
             {:decoder-threshold 1
              ;; -------------------------------
              :max-steps 7 ;; 👈
              ;; ------------------------------
              :stop? (fn [acc next-outcome]
                       (when (< (:confidence next-outcome)
                                0.5)
                         {:stop-reason :low-confidence
                          :success? false}))
              :top-k 1})
           :result-xs
           (map :result)
           (map pyutils/ensure-jvm)))))

'((:o) (:p) (:q) (:r) (:s) (:t) (:u) (:w))


















;; ------------





;;
;; side note:
;; -------------
;;


;;
;; (accidentally did top-k 2, then out comes q b)
;; is something not figured out in the current read version.
;; 'confidence' is high, even though the contribution from q is probably spurious.
;;
;; it would be fixed with a read-threshold
;;
(do (sdm/reset sdm)
    (let [a-kont (:result
                  (sdm/lookup sdm (hdd/clj->vsa* :a) 2 1))]
      (map hdd/cleanup*
           (->> (sdm/converged-lookup-impl
                 sdm
                 a-kont
                 {:decoder-threshold 1
                  :max-steps (count alphabet)
                  :stop? (fn [acc next-outcome]
                           (when (< (:confidence next-outcome)
                                    0.5)
                             {:stop-reason :low-confidence
                              :success? false}))
                  :top-k 1})
                :result-xs
                (map :result)
                (map pyutils/ensure-jvm)))))

'((:q :b)
 (:c)
 (:d)
 (:e)
 (:f)
 (:g)
 (:i)
 (:j)
 (:k)
 (:l)
 (:m)
 (:n)
 (:p)
 (:q)
 (:r)
 (:s)
 (:t)
 (:u)
 (:w)
 (:x)
 (:y)
  (:z))

























;; ------------------------------------------

(defprotocol TrajectoryEngine
  (seed [this])
  (replay [this query]))

(defn ->trajectory-engine
  [{:keys [trajectory-length k-delays]}]
  (let [sdm (sdm/k-fold-sdm {:address-count (long 1e6)
                             :address-density 0.000006
                             :k-delays k-delays
                             :word-length (long 1e4)})]
    (reify
      TrajectoryEngine
        (seed [this]
          (let [trajectory (repeatedly trajectory-length hd/->seed)]
            (sdm/reset sdm)
            (doseq [[addr content]
                      (partition 2 1 trajectory)]
              (sdm/write sdm addr content 1))
            trajectory))
        (replay [this q]
          (sdm/reset sdm)
          (->> (sdm/converged-lookup-impl
                 sdm
                 ;; (reify sdm/SDM
                 ;;   (lookup [this addr n k]
                 ;;     (sdm/lookup
                 ;;      sdm-cleanup
                 ;;      (:result (sdm/lookup sdm addr
                 ;;      n k))
                 ;;      1
                 ;;      1)))
                 (hdd/clj->vsa* q)
                 {:decoder-threshold 1
                  :max-steps trajectory-length
                  :stop? (fn [acc next-outcome]
                           (when (< (:confidence
                                      next-outcome)
                                    0.05)
                             :low-confidence))
                  :top-k 1})
               :result-xs
               (map :result)
               (map pyutils/ensure-jvm))))))

(for [k-delay [5 10 15]]
  (let [trajector (->trajectory-engine {:k-delays k-delay
                                        :trajectory-length
                                          20})
        x (seed trajector)]
    (let [sims (map #(hd/similarity %1 %2)
                 (replay trajector (first x))
                 x)]
      [(every? #(< 0.99 %) sims) (count sims)])))

'([true 20] [true 20] [true 20])


;; - that is with a single round of presentation.

;; Address count is ca 120, which is split on k-delay delay lines.

;;
;; - this is perhaps more similar to the 'scaffold' mem. of Papadimitriou 2023
;; - I see the analogy roughly between delay lines and preallocated scaffold trajectories.
;;
;;

;; ------------------------
;; This memory also has the capability to distinguish higher order sequences.
;;



;; as single order mem:
;; (when k-delay = 1), then first order memory.
(for [k-delay [1 5 10]]
  (let [trajector (->trajectory-engine
                   {:k-delays 1
                    :trajectory-length 50})
        x (seed trajector)]
    (let [sims (map #(hd/similarity %1 %2)
                    (replay trajector (first x))
                    x)]
      [(every? #(< 0.99 %) sims) (count sims)])))
'([true 50] [true 50] [true 50])

;; (k-delay doesn't matter in this context, because there is only 1 thing in memory)
;;

(for [k-delay [5]
      sequence-count [1 10 20 50]]
  (let [trajector (->trajectory-engine
                   {:k-delays 1 :trajectory-length 50})
        _ (doseq [x (range sequence-count)]
            (seed trajector))
        x (seed trajector)]
    (let [sims (map #(hd/similarity %1 %2)
                    (replay trajector (first x))
                    x)]
      [(every? #(< 0.99 %) sims) (count sims)])))

'([true 50] [true 50] [true 50] [true 50])

;; I'm not showing it here rn, but it handles overlapping sequences well.
;;





;; ---------------------------------------------------
;; This is a kind of polychronous bind:
;; -------------------

(do
  (def sdm (sdm/k-fold-sdm {:address-count (long 1e6)
                            :address-density 0.000006
                            :k-delays 2
                            :word-length (long 1e4)}))
  (sdm/reset sdm)
  (sdm/write sdm (hdd/clj->vsa* :a) (hdd/clj->vsa* :a) 1)
  (sdm/write sdm (hdd/clj->vsa* :b) (hdd/clj->vsa* :b) 1)
  (sdm/write sdm (hdd/clj->vsa* :c) (hdd/clj->vsa* :c) 1))

(do
  (sdm/reset sdm)
  (sdm/lookup sdm (hdd/clj->vsa* #{:a :b}) 2 1)
  (def current (sdm/lookup sdm (hdd/clj->vsa* #{:a :b}) 3 1)))
(hdd/cleanup* (pyutils/torch->jvm (:result current)))


;; -----------------------------
;; capacity?
;;

(do (def sdm
      (sdm/k-fold-sdm {:address-count (long 1e6)
                       :address-density 0.000006
                       :k-delays 2
                       :word-length (long 1e4)}))
    (sdm/reset sdm)
    (sdm/write sdm
               (hdd/clj->vsa* :left)
               (hdd/clj->vsa* :left)
               1)
    (sdm/write sdm
               (hdd/clj->vsa* :right)
               (hdd/clj->vsa* :right)
               1)
    (sdm/write sdm
               (hdd/clj->vsa* :right)
               (hdd/clj->vsa* :right)
               1)
    (sdm/write sdm
               (hdd/clj->vsa* [:*> :left :right :right])
               (hdd/clj->vsa* [:*> :left :right :right])
               1))


(do (sdm/reset sdm)
    (def query (hdd/clj->vsa* [:*> :left :right :right]))
    (let [x1 (pyutils/torch->jvm
              (:result (sdm/lookup sdm
                                   (hdd/clj->vsa*
                                    #{:left :right})
                                   2
                                   1)))
          x2 (pyutils/torch->jvm
              (:result (sdm/lookup sdm
                                   (hdd/clj->vsa*
                                    #{:left :right})
                                   2
                                   1)))
          x3 (pyutils/torch->jvm
              (:result (sdm/lookup sdm
                                   (hdd/clj->vsa*
                                    #{:left :right})
                                   2
                                   1)))
          xr (pyutils/torch->jvm
              (:result (sdm/lookup sdm
                                   (hdd/clj->vsa*
                                    #{:left :right})
                                   3
                                   1)))]
      ;; [x1 x2 x3]
      (map hdd/cleanup* [x1 x2 x3])
      (map hdd/cleanup*
           (map-indexed (fn [i x]
                          (hd/unbind query (hd/permute-n x i)))
                        [x1 x2 x3]))))

(hdd/cleanup* (pyutils/torch->jvm (:result current)))


















;; ----------------------------------------


;; Lit
;;
;; - https://arxiv.org/abs/2306.03812
;; - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5628090/
;; - https://www.sciencedirect.com/science/article/pii/S009286742031388X
;;
;; - https://pubmed.ncbi.nlm.nih.gov/15105494/
