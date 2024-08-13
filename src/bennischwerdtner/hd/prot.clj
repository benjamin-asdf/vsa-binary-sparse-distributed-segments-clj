(ns bennischwerdtner.hd.prot)

(defprotocol ItemMemory
  (m-clj->vsa [this item])
  (m-cleanup-verbose [this q]
    [this q threshold])
  (m-cleanup [this q])
  (m-cleanup* [this q]
    [this q threshold]))
