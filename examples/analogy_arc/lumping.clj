(ns analogy-arc.lumping
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
   [bennischwerdtner.sdm.sdm :as sdm]
   [bennischwerdtner.hd.codebook-item-memory :as codebook]
   [bennischwerdtner.hd.ui.audio :as audio]
   [bennischwerdtner.hd.data :as hdd]))

(alter-var-root
 #'hdd/*item-memory*
 (constantly (codebook/codebook-item-memory 1000)))

;; ---
;; The lump:
;;
;; ---

;; retina:
;;
;;    [p-1, p0, p1 ]
;;
;; effector layer:
;;
;;    [ p-1, p0, p1  ]
;;

(def retina-register
  [(hdd/clj->vsa* :j)
   (hdd/clj->vsa* :m)
   (hdd/clj->vsa* :m)])

(def retina-effector-address (into [] (range 3)))

(def motor-register
  (map hdd/clj->vsa* retina-effector-address))

(def association-register
  (into [] (map hd/bind retina-register motor-register)))

(hdd/cleanup*
 (hd/unbind
  (apply hdd/set association-register)
  (apply hdd/set motor-register)))
'(:j :m)

(hd/unbind
  (apply hdd/set association-register)
  (apply hdd/set motor-register))


;;
;; :m tolerates a range of movement
;; [p0, p1]
;;



(def sdm (sdm/->sdm {:address-count (long 1e6)
                     :address-density 0.000003
                     :word-length (long 1e4)}))

(map (fn [m s]
       (sdm/write sdm m s 1))
     motor-register retina-register)











(sdm/write sdm (hdd/clj->vsa* :a) (hdd/clj->vsa* :a) 1)

(hdd/cleanup
 (pyutils/ensure-jvm
  (:result (sdm/lookup sdm (hdd/clj->vsa* :a) 1 1))))
