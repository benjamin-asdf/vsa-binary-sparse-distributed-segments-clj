(ns binding-in-superposition
  (:require [tech.v3.datatype.functional :as f]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
            [tech.v3.datatype.bitmap :as bitmap]
            [fastmath.random :as fm.rand]
            [fastmath.core :as fm]
            [bennischwerdtner.hd.binary-sparse-segmented :as
             hd]
            [tech.v3.datatype.unary-pred :as unary-pred]
            [tech.v3.datatype.argops :as dtype-argops]
            [bennischwerdtner.hd.data :as hdd]))



;; this is the crossproduct the 2 sets
(hd/bind (hdd/clj->vsa* #{:a :b :c})
         (hdd/clj->vsa* #{:x :y :z}))

(hdd/cleanup*
 (hd/unbind
  (hd/bind (hdd/clj->vsa* #{:a :b :c})
           (hdd/clj->vsa* #{:x :y :z}))
  (hdd/clj->vsa* :y)))
'(:b :a :c)



(hdd/cleanup*
 ;; obviously roughly #{:x :y :z}
 (hd/unbind
  (hd/bind
   (hdd/clj->vsa* #{:a :b :c})
   (hdd/clj->vsa* #{:x :y :z}))
  (hdd/clj->vsa* :a)))
'(:y :z :x)

(hdd/cleanup*
 (hd/unbind
  ;; ~ {:a :x}, the only thing in the
  ;; crossproduct that is the same
  (hdd/intersection
   (hd/bind (hdd/clj->vsa* #{:a :b :c})
            (hdd/clj->vsa* #{:x :y :z}))
   (hd/bind (hdd/clj->vsa* #{:a :foo :bar})
            (hdd/clj->vsa* #{:x :hurr :durr})))
  (hdd/clj->vsa* :a)))
'(:x)

(hdd/clj->vsa* :a)
#tech.v3.tensor<int8>[10000]
[0 0 0 ... 0 0 0]

(hdd/cleanup* *1)
'(:a)


(=
 (hdd/clj->vsa* #{:a :b :c})
 (hdd/set
  (hdd/clj->vsa* :a)
  (hdd/clj->vsa* :b)
  (hdd/clj->vsa* :c))
 (hd/superposition
  (hdd/clj->vsa* :a)
  (hdd/clj->vsa* :b)
  (hdd/clj->vsa* :c)))
true

(hdd/cleanup*
 (hd/unbind
  (hdd/clj->vsa* {:a :x :b :y :c :z})
  (hdd/clj->vsa* :a)))
(:x)


(=
 (hdd/clj->vsa* {:a :x :b :y :c :z})
 (hd/superposition
  (hd/bind (hdd/clj->vsa* :a)
           (hdd/clj->vsa* :x))
  (hd/bind (hdd/clj->vsa* :b)
           (hdd/clj->vsa* :y))
  (hd/bind (hdd/clj->vsa* :c)
           (hdd/clj->vsa* :z))))
true






;; -------------------

(def sun (hdd/clj->vsa* #{:yellow :warm :space-object :round}))
(def banana (hdd/clj->vsa* #{:yellow :food :tasty :fruit :banana-shaped}))

(hdd/cleanup* (hdd/intersection sun banana))
'(:yellow)

;; -------------------

(def fruit-domain (hdd/clj->vsa*
                   {:lemon :yellow :apple :red :orange :orange}))


;; What fruit has the color that sun and banana share? (or something like that)

(hdd/cleanup*
 (hd/unbind
  fruit-domain
  (hdd/intersection sun banana)))
'(:lemon)


(hdd/cleanup*
 (hdd/difference
  (hdd/clj->vsa* #{:ice :cold :hard :water})
  (hdd/clj->vsa* #{:cold :hard :stony :earth})))
'(:ice :water)
