(ns bennischwerdtner.hd.prot)

(defprotocol ItemMemory
  (m-clj->vsa [this item])
  (m-cleanup-verbose [this q]
    [this q threshold])
  (m-cleanup [this q])
  (m-cleanup* [this q]
    [this q threshold]))





;; (defprotocol HDV
;;   (unbind [this & other])
;;   (superposition [this & other])
;;   (hd-drop [this amount])
;;   (permute-n [this n])
;;   (hd-bind [this]
;;     [this & others]))

;; (extend-type Object
;;   HDV
;;   (hd-bind
;;     ([this] this)
;;     ([this & others] others)))







;; ---------------
;; hd/bind jvmhv pythonhv
;; -> jvmhv


;; hd/similarity
;; ...
;;

;; hd/bind pythonhv pythonhv
;; -> pythonhv
