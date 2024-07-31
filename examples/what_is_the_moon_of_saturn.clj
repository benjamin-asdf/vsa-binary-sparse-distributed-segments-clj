(ns what-is-the-moon-of-saturn
  (:require
   [tech.v3.datatype.functional :as f]
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

(def earth
  ;; moon the prototype and moon the filler
  ;; could be [:moon :moon]
  (hdd/->directed-edge (hdd/clj->vsa* [:moon :luna])))

(def saturn
  (hdd/directed-graph (hdd/clj->vsa*
                       [[:adjective :saturnian]
                        [:adjective :cronian]
                        [:adjective :kronian]
                        [:rings true]
                        [:moon :mimas]
                        [:moon :enceladus]
                        [:moon :tethys]
                        [:moon :dione]
                        [:moon :rhea]
                        [:moon :titan]
                        [:moon :iapetus]])))


;; let's say you know luna and you want to know what that is in saturn domain

(hdd/cleanup*
 (hdd/edge->destination
  saturn
  (hdd/edge->source earth (hdd/clj->vsa :luna))))
'(:rhea :iapetus :tethys :dione :mimas :titan :enceladus)


;; 0. "What is the dollar in mexico?" kind of things work in general.
;; 1. the moon of saturn is a superposition 7 things
;; 2. Saturn is a composite datastructure, yet we pretend it's one element 'edge->destination'
;;    works on an edge element, and the superposition of elements




;; partial expansion:
;; ----------------

(hdd/cleanup*
 (hdd/edge->destination
  saturn
  (hdd/edge->source
   (hdd/->directed-edge (hdd/clj->vsa* [:moon :luna]))
   (hdd/clj->vsa :moon))))


(hdd/cleanup* (hd/unbind
                ;; earh
                (hd/bind (hdd/clj->vsa* :moon)
                         (hd/permute (hdd/clj->vsa* :moon)))
                (hd/permute (hdd/clj->vsa* :moon))))
'(:moon)

;; I am not expanding 'saturn' now for brevity.
;; But it is a set of directed edges.


(hdd/cleanup*
 (hd/permute-inverse
  (hd/unbind
   saturn
   ;; moon
   (hd/unbind
    ;; earth
    (hd/bind
     (hdd/clj->vsa* :moon)
     (hd/permute (hdd/clj->vsa* :moon)))
    (hd/permute (hdd/clj->vsa* :moon))))))

;; ------------------------------------------
