(ns iterative-block-code-factorizer
  (:require
    [bennischwerdtner.pyutils :as pyutils]
    [bennischwerdtner.hd.prot :as prot]
    [tech.v3.datatype.functional :as f]
    [tech.v3.datatype :as dtype]
    [tech.v3.tensor :as dtt]
    [tech.v3.datatype.bitmap :as bitmap]
    [fastmath.random :as fm.rand]
    [fastmath.core :as fm]
    [bennischwerdtner.hd.binary-sparse-segmented :as hd]
    [tech.v3.datatype.unary-pred :as unary-pred]
    [tech.v3.datatype.argops :as dtype-argops]
    [bennischwerdtner.hd.data :as hdd]
    [libpython-clj2.require :refer [require-python]]
    [libpython-clj2.python.ffi :as py-ffi]
    [libpython-clj2.python :refer [py. py.. py.-] :as py]))



;;
;; paper:
;; https://arxiv.org/abs/2303.13957
;; Factorizers for Distributed Sparse Block Codes
;;

;;
;; I tried playing around with resonator networks, and indeed the block sparse vectors don't work well.
;; One issue was that I don't really have something to bind/unbind with denser vectors.
;; So info quickly is lost, if unbind with a superposition (which is the initial step in a resonator network).
;; Also the unit vector as identity element didn't turn out great, because in the block sparse encoding, the superposition that
;; includes the unit vector basically does an identiy unbind. (So the basic mechanism of resonator network is disrupted).
;;
;;
;;
;;
;;

;;
;;  - each factor f has a codeboook Mf codevectors
;;  - there are Mf1 * Mf2 * ... many possible combinations to be searched
;;




;; Generilzed sparse block codes (GSBCs)
;;
;; Like SBCs, devide the Dp-dimensional into B blockss of equal length L = Dp/B
;; Called N, segment-count and segment-length here.
;;
;; - don't restrict blocks to be binary or sparse
;; - require: elements are real numbers and each block has a l1-norm
;; - SBCs are valid GSBCs
;; -
;;

;; the say elementwise difference?
;; - they take the max index
;;
;;
;;
;; - l-infinity norm: Only the largest element in the vector counts
;;
;;

;; given 2 notebooks, and a product vector p from the notebooks,
;; estimate the factors so that
;; p = x1 ⊗ x2


;; Step 0:
;; bundle codebooks as initial estimates
;; (its the superposition of all possible)
;;

;; Step 1:
;; Unbinding (blockwise circular correlation)
;;
;;
;; estimate-x1-next = p ⊘ estimate-x1
;; ...
;;

;; Step 2:
;; Similarity search
;; query the associative memory with the unbound factor estimates
;; l-infinite syeilds a vecotr of similarity scores
;;
;; a-f(t)[i]  = s-similarity(x-estimate(t), codebook-i)
;;
;;

;; Step 3:
;; Sparse activation and conditional random sampling:
;;
;; zero out activations below the threshold T
;;
;;
;; - if it is all zeros, then randomly activate
;; - A-many values set to 1/A
;;

;; Step 4:
;; Weighted bundling:
;;
;; the next factor estimate as the normalized wieghted bundling of the factors codevectors:
;;
;;  codebook[activations]  / sum of activations
;;

;; Step 5:
;; Convergence detection
;; iteratively decode until it converges or a maximum number of iterations (N) is reached.
;;
;; N = count of brute force
;;
;; stop when a the similarity vectors contain an element that exceeds a detteciton thershold value Tc
;;

;;
