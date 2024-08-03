(ns what-is-the-heart-of-the-solar-system
  (:require [tech.v3.datatype.functional :as f]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
            [tech.v3.datatype.bitmap :as bitmap]
            [fastmath.random :as fm.rand]
            [fastmath.core :as fm]
            [bennischwerdtner.sdm.sdm :as sdm]
            [bennischwerdtner.hd.binary-sparse-segmented :as
             hd]
            [bennischwerdtner.pyutils :as pyutils]
            [tech.v3.datatype.unary-pred :as unary-pred]
            [tech.v3.datatype.argops :as dtype-argops]
            [bennischwerdtner.hd.data :as hdd]))

;; wip

(hdd/cleanup*
 (hdd/intersection
  (hd/thin (hdd/clj->vsa* #{:x :a}))
  (hd/thin (hdd/clj->vsa* #{:x :c}))))

(hdd/cleanup*
 (hdd/difference
  (hdd/intersection
   (hd/thin (hdd/clj->vsa* #{:x :a}))
   (hd/thin (hdd/clj->vsa* #{:x :c})))
  (hdd/clj->vsa* :x)))


(hdd/cleanup*
 (hdd/difference
  (hdd/clj->vsa* #{:x :a})
  (hdd/clj->vsa* :x)))
