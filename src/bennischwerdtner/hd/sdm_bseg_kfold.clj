(ns bennischwerdtner.hd.sdm-bseg-kfold
  (:require
   [bennischwerdtner.hd.binary-sparse-segmented :as hd]
   [clojure.test :as t]
   [tech.v3.datatype.functional :as f]
   [tech.v3.parallel.for :as pf]
   [tech.v3.tensor :as dtt]
   [tech.v3.datatype.bitmap :as bitmap]
   [fastmath.random :as fm.rand]
   [fastmath.core :as fm]
   [tech.v3.datatype.unary-pred :as unary-pred]
   [tech.v3.datatype.argops :as dtype-argops]))

;; Jaeckel, L.A. 1989a. An Alternative Design for a Sparse Distributed Memory.
;; Report RIACS TR 89.28, Research Institute for Advanced Computer Science,
;; NASA Ames Research Center.
;;
;; Jaeckel, L.A. 1989b. A Class of Designs for a Sparse Distributed Memory. Report
;; RIACS TR 89.30, Research Institute for Advanced Computer Science, NASA
;; Ames Research Center.

;; But here, I make an example of binary sparse segmented (hypervector) words


;; -----------------
;; Idea:
;; - Parallel fibers have slow transduction speed
;; - Cerebellum has been suggested to be a ms-range timer (Braitenberg 1967)
;;   https://www.sciencedirect.com/science/article/abs/pii/S0079612308609711
;; - The logic of k-fold higher order sequence encoding works via delays (Kanerva 1988)
;; - Delaying the parallel fiber input might implement a k-fold sequence delay,
;;   implementing higher order sequencing
;;
;; This is extremely satisfying to me, because it unifies Braitenberg and Albus+Marr theories of cerebellar computation,
;; which traditionally looked like alternatives to each other.
;;
;; This would solve the issue of why parallel fibres did not have an evolutionary driver for fast transduction.
;; Sdm design (implicitly) assumes that the count of pf->purkinje synapses is maximized,
;; and that that thin diameter of parallel fibres comes from the geometry of packing as many pf's as possible.
;;
;; But slow speed, hence delays would be neccessary for encoding sequences in a k-fold design.
;; If a k-fold delay-line sdm hypothesis (hereby named) of cerebellar cortex is true,
;; the parallel fibres would have evolutionary drivers to make delays, hence stay thin and umyelinated
;;

;; - in this version, reading and writing at the content matrix would have variable amounts of delays.
;; Assume
;; - discrete time steps
;; - an address location has bit locations for t0, t1 ... up to t-range (e.g. t-range = 7)
;; - alternatively each location could have a delay associated with it, since you read from multiple locations, some would
;;   be delay=0 and so forth (this only essentially overlaps with the biological model).
;; - to write a sequence [a b c], one would decode addresses for a,b,c. Then write the bit locations a-addresses[t=0],
;;   then write the bit locations b-addresses[t=0] union a-addresses[t=1] and so forth.
;;


(def counter-range [0 15])
;; perhaps 7 plus or minus 2
(def delay-max 5)
