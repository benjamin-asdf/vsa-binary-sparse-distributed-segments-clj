;;
;; I put my attention somwhere else for the moment
;; Kinda shows that you *can* train a k fold sdm to represent seuquences / likelyhoods
;;
;;
(ns shakespear
  (:require
   [clojure.set]
   [bennischwerdtner.hd.binary-sparse-segmented :as hd]
   [tech.v3.datatype :as dtype]
   [tech.v3.parallel.for :as pfor]
   [tech.v3.datatype.argops :as dtype-argops]
   [tech.v3.tensor :as dtt]
   [bennischwerdtner.sdm.sdm :as sdm]
   [tech.v3.datatype.functional :as f]))

(require '[sequence-processor :as hl])

(defonce tiny-sp-text (slurp "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"))

(alter-var-root
   #'hd/default-opts
   #(merge %
           (let [dimensions (long 1e4)
                 segment-count 20]
             {:bsdc-seg/N dimensions
              :bsdc-seg/segment-count segment-count
              :bsdc-seg/segment-length
              (/ dimensions segment-count)})))

(def k-fold-sdm
  (sdm/->k-fold-memory
    {:address-count (long 1e5)
     :address-density 0.00003
     :k-delays 6
     :stop? (fn [acc next-outcome]
              (when (< (:confidence next-outcome) 0.05)
                :low-confidence))
     :word-length (long 1e4)}))

(do
  (def vocab (into #{} tiny-sp-text))
  (defonce encode
    (into {}
          (map (juxt identity (fn [_] (hd/->seed))))
          vocab))
  (def encode-1 (clojure.set/map-invert encode))
  (def decode
    (comp
      (fn [tens]
        (let [m (into [] (vals encode))
              similarities
                (into [] (pmap #(hd/similarity % tens) m))]
          similarities
          (when (seq similarities)
            (let [argmax (dtype-argops/argmax similarities)]
              (when (<= 0.9 (similarities argmax))
                (encode-1 (m argmax)))))))
      sdm/ensure-jvm)))

(comment
  (decode (encode \I))
  \I)

(comment
  (apply str
         (->> (do (sdm/reset k-fold-sdm)
                  (sdm/lookup-xs k-fold-sdm (encode \F) 1))
              :result-xs
              (map :result)
              (map decode)))
  "First Ce")

(defn shakespear-v0
  [start-character]
  (let [{:keys [result-xs]}
        (do (sdm/reset k-fold-sdm)
            (sdm/lookup-xs k-fold-sdm (encode start-character) 1))]
    (for [x result-xs]
      (decode (:result x)))))


(comment
  (apply str (shakespear-v0 \F))
  "First Ce")


(comment


  (time
   (doseq
       [txs
        (take 200 (partition-all 7 tiny-sp-text))]
       (do
         (sdm/reset k-fold-sdm)
         (sdm/write-xs! k-fold-sdm (map encode txs) 1))))

  ;;
  ;; would be 6 hours to train on the the whole text
  ;;

  (apply str (shakespear-v0 \F))
  "Fire  oo"

  (apply str (shakespear-v0 \U))
  "U th  oe"

  (apply str (shakespear-v0 (rand-nth (into [] (keys encode)))))
  "On:\n\n\n\n\n"
  "You th  "
  "'t th  o"
  "$ th  oe"
  "\n\n\n\n\n\n\n\n"
  "Ds th  o"
  "Se t "

  ;; newline is associated with itself lol.


  )

(comment

  (time
   (doseq
       [txs
        (take 500 (drop 700 (partition-all 7 (remove #{\space \newline} tiny-sp-text))))]
       (do
         (sdm/reset k-fold-sdm)
         (sdm/write-xs! k-fold-sdm (map encode txs) 1))))

  ;;
  ;; would be 6 hours to train on the the whole text
  ;;
  (for [n (range 10)]
    (apply str (shakespear-v0 (rand-nth (into [] (keys encode))))))

  ("Hathenet"
   "MENENIUS"
   "Nndither"
   "Lenhere"
   "Youreren"
   "Theene"
   "peenee"
   "K"
   "Q"
   "reneere")

  ;; e starts to dominate the landscape now

  (for [n (range 10)]
    (apply str (shakespear-v0 (rand-nth (into [] (keys encode))))))

  ("Leresee"
   "lloure"
   "zustheth"
   "whereser"
   "MENENIUS"
   "Goureath"
   ",thethe"
   "\neresee"
   "qustheth"
   "Yeresee"))
