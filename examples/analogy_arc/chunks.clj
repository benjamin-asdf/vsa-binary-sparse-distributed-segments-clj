(ns analogy-arc.chunks
  (:require
    [tech.v3.datatype.functional :as f]
    [tech.v3.datatype :as dtype]
    [tech.v3.tensor :as dtt]
    [tech.v3.datatype.bitmap :as bitmap]
    [fastmath.random :as fm.rand]
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
;;
;;

(def alphabet
  (into []
        (map (comp keyword str char)
             (range (int \a) (inc (int \z))))))

(def world
  (into {}
        (concat (map (fn [a b] [[a :right] b])
                  alphabet
                  (drop 1 alphabet))
                (map (fn [a b] [[a :left] b])
                  (reverse alphabet)
                  (drop 1 (reverse alphabet))))))


;; --------------------------------------------------
;; How to move in concept spaces?
;;
;; - effectors
;;
;;
;; I: Getting Around
;;
;;

(defprotocol Mover
  (effect [this world]))








;;
;;
;;
;;
;;                                          +--------+
;;                                          |        |
;;                              +-----------+        | trajectory memory
;;                              |           +--------+
;;                              |
;;                              |
;;                              |     seed: create new trajectory (neuronal word length wide)
;;                              |
;;                              |
;;                              v
;;                 +--------------------+                                                          +----------------+
;;                 |                    | focus  neuronal-word-length wide                         |                |
;;                 +--------------+-----+                                                          +----------------+
;;                                |                                                       |            sensoric activator
;;                                |        (~ population vectors)                         |
;;                                |                                                       |
;;                                | mover                                                 |
;;                                |                                                       |
;;                                |                                                       |
;;                                |             +---------------------------------------> |
;;                                v             |
;;                                              |                                        sensory surface
;;                     world  s1, s2,s3,        |
;;
;;


;; ---------------------------------------------------

;; V1: World Finder
;; Cognitive Architecture:
;;
;;
;;
;;
;;
;;
;;
;;
;;
;;
;;                            +--------------------------+
;;                            |                          |
;;                            |                          | focus
;;                            +--------------------------+
;;
;;
;;
;;                             +------------------------+
;;                             |    ^                   | sensoric surface
;;                             +----+-------------------+
;;                                  |
;;                                  |
;;                                  |
;;                                world



;; ---------------------------

;;
;; 1. activator - global address decoder input word
;;
;; activator    -> [ sdm, sdm, sdm, ... ] in parallel (conceptron)
;;
;;                 non liniearity:
;;
;; 2. focus     - global output state
;;

(def state
  {:activator (hd/->seed)
   :conceptron [(sdm/->sdm {:address-count (long 1e6)
                            :address-density 0.000003
                            :word-length (long 1e4)})
                (sdm/->sdm {:address-count (long 1e6)
                            :address-density 0.000003
                            :word-length (long 1e4)})
                (sdm/->sdm {:address-count (long 1e6)
                            :address-density 0.000003
                            :word-length (long 1e4)})]
   :focus (hd/->seed)})

(def commitors
  [{:right (hdd/clj->vsa* :right)}
   {:left (hdd/clj->vsa* :left)}])


;; -----------

(defn commitment-structure [{:keys [focus comitment]}]

  )


;; -----
;;


(->
 state
 (assoc :activator))



(hd/similarity
 (hdd/clj->vsa* [:*> :a :b :c :d :e :f :g])
 (hd/unbind (hdd/clj->vsa* :a))
 (hdd/clj->vsa* [:> [:*> :b :c :d :e :f :g]]))

(def focus (hdd/clj->vsa* [:*> :a :right :b :right :c :right :d]))
(def action-structure
  (hd/unbind
   focus
   (hdd/clj->vsa*
    [:*
     [:> :a 0]
     [:> :b 2]
     [:> :c 4]
     [:> :c 6]])))

(map
 hdd/cleanup*
 (take 7 (iterate
          (fn [x] (hd/permute-inverse x))
          (hd/unbind focus action-structure))))

(hd/similarity
 (hdd/clj->vsa* [:> :right])
 (hd/unbind focus action-structure))



(def focus (hdd/clj->vsa* [:*> :a :right :b]))

(def action-structure
  (hd/unbind focus
             (hdd/clj->vsa*
              [:*
               [:> :a 0]
               [:> :b 2]])))

(map hdd/cleanup*
  (take 7
        (iterate (fn [x] (hd/permute-inverse x))
                 (hd/unbind focus
                            (hdd/clj->vsa* [:* [:> :a 0]
                                            [:> :b 2]])))))

(hd/similarity
 (hdd/clj->vsa* [:> :right])
 (hd/unbind focus action-structure))

(hdd/clj->vsa* [:*> :a :right :a])

(def sdm
  (sdm/->sdm {:address-count (long 1e6)
              :address-density 0.000003
              :word-length (long 1e4)}))

(doseq [sdm (:conceptron state)]
  (doall
   (map
    (fn [x] (sdm/write sdm x x 1))
    (map hdd/clj->vsa* alphabet))))

;; -----

(hdd/clj->vsa*
 [:?=
  [:* :a :x]
  [:*
   [:+ :a [:-- :b 0.5]]
   [:+ :x [:-- :y 0.5]]]])

(hdd/clj->vsa*
 [:?=
  [:* :a :x :1]
  [:*
   [:+ :a :b :c]
   [:+ :x :y :z]
   [:+ :0 :1 :2]]])

;;
(* 60 18)
;;
