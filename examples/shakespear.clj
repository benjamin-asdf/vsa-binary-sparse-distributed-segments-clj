(ns shakespear
  (:require
   [bennischwerdtner.hd.binary-sparse-segmented :as
    hd]
   [tech.v3.datatype :as dtype]
   [tech.v3.parallel.for :as pfor]
   [tech.v3.datatype.argops :as dtype-argops]
   [tech.v3.datatype.functional :as f]))

(require '[sequence-processor :as hl])

(defonce tiny-sp-text (slurp "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"))

(def db (atom (hd/->seed)))

;; https://paperswithcode.com/paper/cognitive-modeling-and-learning-with-sparse#code

;; binding a window with sequence markers,
;; creating a single high dimensional representation

;; (partition 3 1 [:a :b :c :d])

(def items [:a :b :c :d])

(defn bundle-seq
  [db [a b c]]
  (apply
   hd/bundle
   (concat
    [db]
    (map (fn [position e]
           (hl/bind (hl/sequence-marker
                     position)
                    (hl/clj->vsa e)))
         [-1 0 1]
         [a b c]))))

(hl/cleanup*
  (hd/unbind
    (hd/thin
      (reduce bundle-seq
        (hd/->seed)
        (partition 3 1 (concat [:sos] items [:eos]))))
    (hl/sequence-marker 0)))

(hd/similarity
 (hl/sequence-marker -1)
 (hd/unbind
  (hd/thin
   (reduce bundle-seq
           (hd/->seed)
           (partition 3 1 (concat [:sos] items [:eos]))))
  (hl/clj->vsa :sos)))


(hl/cleanup*
 (hd/unbind
  (hd/thin
   (reduce bundle-seq
           (hd/->seed)
           (partition 3 1 (concat [:sos] items [:eos]))))
  (hd/bundle
   (hl/clj->vsa :sos)
   (hl/sequence-marker 0))))
