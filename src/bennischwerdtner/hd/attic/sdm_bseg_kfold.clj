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

;;
;; ----------------------------
;; k-fold sdm
;; ----------------------------
;;
;; Initial idea:
;; - Parallel fibers have slow transduction speed
;; - Cerebellum has been suggested to be a ms-range timer (Braitenberg 1967)
;;   https://www.sciencedirect.com/science/article/abs/pii/S0079612308609711
;; - The logic of k-fold higher order sequence encoding works via delays (Kanerva 1988)
;; - Delaying the parallel fiber input might implement a k-fold sequence delay,
;;   implementing higher order sequencing
;;
;;
;;  ... as Kanerva points out, a k-fold memory can be achieved by delaying either address decoder input or output
;;      randomly proportianlly.
;;      The idea here was to delay address decoder output, but spread out at C segments.
;;      Perhaps this biologcially more plausible, but random delays for addresses sound easier to implement
;;
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
;;   be delay=0 and so forth (this would essentially make a k-fold design. But would not fit the idea that the pf's yield the delay)
;; - to write a sequence [a b c], one would decode addresses for a,b,c. Then write the bit locations a-addresses[t=0],
;;   then write the bit locations b-addresses[t=0] union a-addresses[t=1] and so forth.
;;

;;
;;
;;                               w
;;
;;                               |
;;                               |
;;                               v
;;              y
;;            +----+   +------+------+------+
;; A ---->    | 1  +--->      |      |      |
;;            |    |   |      |  C   |      |
;;            | 1  +--->      |      |      |
;;            |    |   |      |      |      |
;;            |    |   |      |      |      |
;;            |    |   |      |      |      |
;;            |    |   |      |      |      |
;;            |    |   |      |      |      |
;;            +----+   +------+------+------+
;;                        ^
;;                        |
;;                   -----+
;;            delay segments         , ... k-delay-segments = 5
;;
;;
;;
;;                        |       |
;;                        |t1     | t2   ... (k-delays = 5)
;;                        v       v
;;                     +------+--------------+
;;           t1        | 1  1 |              |
;;                     +------+--------------+
;;
;;
;;                     +------+-------+------+
;;           t2        | 1  1 | 1  1  |      |
;;                     +------+-------+------+
;;


;;
;; - conceptually, in order to read a full word you need `k-delays` timesteps
;; - write:
;;   - Imagine you write with `b`, but some address locations of `a` are on, in fact "address-locations[a,tn]",
;;     where tn is the time n time steps since writing `a`.
;;
;; - the input patterns of mossy fibers would be interesting to know
;; - for instance, if the same mossy fibers are on with frequency m-freq. M-freq could correspond to the temporal extend of one delay-segment
;; - (i.e. the conduction speed of parallel fibers * m-frequence-time = segment length),
;; - if this would be true, then we would 'write' a `k-delay` times in a write operation
;; - and the write operation time might equal to k-delay * delay-time == the time to read a full word
;; (note this also depends on the integration times of the neurons / synapses involved)
;;
;; To write address a:
;;
;; Fast version:
;; 1. decode address locations             t0
;; 2. Activate address locations k = 0,    t0
;; 3. increment bit counters               t0
;; 4.
;;
;;
;;
;;
;;
;; - read:
;;   - Imagine you read with address `a`. At the first time step, you get mostly a parts,
;;     at the second, you get a mix of a and b and so forth.
;;
;;
;;
;; - since we localize the delay at the parallel fibers, the whole address decoder is allowed to stay the same
;;
