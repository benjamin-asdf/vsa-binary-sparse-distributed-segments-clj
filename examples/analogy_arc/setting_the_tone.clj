(ns analogy-arc.setting-the-tone
  (:require
   [tech.v3.datatype.functional :as f]
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


;; The Problem:
;; ---------------------------
;;
;;
;; -
;;


;;

;; 'Setting the tone' basic idea:
;;
;;
;; ----------------
;; A musical encoding looks like a way to make hierarchical sequences. Intuitively, if some piece of music is played,
;; one already knows the continuation vaguely. It's a constraint to some degree, not random and not predetermined.
;; This 'sets the tone' for the message to follow, which can then be nested or terminal sequences.
;; (This stuff is empirical neuroscience by Buszaki!).
;; -----------------
;;
