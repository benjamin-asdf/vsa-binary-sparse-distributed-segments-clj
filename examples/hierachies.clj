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






























;; ----------------------------------------


;; Lit
;;
;; - https://arxiv.org/abs/2306.03812
;; - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5628090/
;; - https://www.sciencedirect.com/science/article/pii/S009286742031388X
;;
;; - https://pubmed.ncbi.nlm.nih.gov/15105494/
