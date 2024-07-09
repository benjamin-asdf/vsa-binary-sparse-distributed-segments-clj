(ns bennischwerdtner.sdm.sdm
  (:require
    [bennischwerdtner.hd.binary-sparse-segmented :as hd]
    [tech.v3.datatype.functional :as f]
    [tech.v3.datatype :as dtype]
    [tech.v3.tensor :as dtt]
    [tech.v3.datatype.bitmap :as bitmap]
    [fastmath.random :as fm.rand]
    [fastmath.core :as fm]
    [tech.v3.datatype.unary-pred :as unary-pred]
    [tech.v3.datatype.argops :as dtype-argops]
    [libpython-clj2.require :refer [require-python]]
    [libpython-clj2.python :refer [py. py.. py.-] :as py]))

;;
;; This is a sparse distributed memory for binary sparse segmented hypervectors
;;
;;
;;

;; Adapted from Kanerva 1993[1]:
;; And mapping to neurophysiology of cerebellum
;;
;;
;;
;;         ADDRESS REGISTER                             WORD IN REGISTER
;;
;;            N = 10.000                                 U = N = 10.000
;;        +--------------+                           +------------------+
;;        | x            |                           |                  |
;;        +------+-------+                           +----------+-------+
;;               |                                              |          'mossy fibers'
;;               |                                              |
;;               |      N       d              y                |      U
;;        +------v-------+    +-+             +-+    +----------v-------+
;;        |              |    |3|             |1+---->                  |
;;        |              |    |0|             |0|    |                  |
;;        |      A       +--> |2| --------->  |1|---->        C         |
;;        | M hard addr. |    |0|  threshold  |0|    |   M x U counters |
;;      M |              |    |1|             |0|  M |                  |
;;        +--------------+    +-+             +-+    +---------+--------+  'parallel fiber -> purkinje'
;;                                                             |
;;                                             |               |
;;                             |               |               v
;;                             |               |      +-----------------+
;;                             |               |   S  |                 |  sums
;;                             v               |      +--+-----+----+---+  'purkinje inputs'
;;                        address overlap      |         |     |    |
;;                         ~d of [1,2]         |         |     |    |
;;                                             |         v     v    v
;;                         activations    <----+                             top-k per segment
;;                         ( 2 <= d)
;;                                                    +-----+----+------+
;;                                                 z  |     |    |      |   'purkinje activations', or downstream purkinje reader
;;                                                    +-----+----+------+
;;                                                      s1    s2, ... segment-count
;;
;;
;;                                                    word out register
;;
;;                                                   10.000 bits = segment-count * segment-length
;;
;;
;;
;; A - Address Matrix                      'mossy fibers -> granule cells synapses'
;; M - hard-locations-count                'granule cell count'
;; y - address-activations                 'granule cell activations'
;; C - Content Matrix                      'parallel fibers -> purkinje synapses'
;;
;;
;;
;;
;; Address decoder
;; ----------------
;;
;; - M hard address locations of N width
;; - address locations are sparse 0s and 1s with address-density << 1
;; - Each address location models one granule cell, granule cells outnumber mossy fibers (their inputs) 200 to 1
;;   (https://en.wikipedia.org/wiki/Cerebellar_granule_cell#:~:text=Granule%20cells%20receive%20all%20of,a%20much%20more%20expansive%20way.)
;;
;;
;; Since addreses are not dense, this is a 'intermediate design' (Jaeckel, L.A. 1989b)
;;
;;
;; Storage / Content Matrix `C`
;; -----------------
;;
;; - M x U counters
;; - here, U == N, allowing for auto association. I.e. address-word == input-word.
;; - Different from Kanerva 1993, where each location is in range {-15...15},
;;   here range is {0..`counter-max`}, counter-max = 100 (?)
;;
;; Read (y):
;; - sum up counters of the y activated address content locations of C (-> S)
;; - In Kanerva 1993: take the sign of the sums
;; - Here, take top-k (`read-k`) non zero bits per word segment
;; - output vector `z` has read-k * segment-count non zero bits
;; - when `read-k` == 1, then `z` is maximally sparse [[hd/maximally-sparse?]]
;;
;;
;; Write (y, input-word):
;;
;; - For each active location, increment the in C where input-word has a non-zero bit
;; - clamp it to the counter range
;;
;;














(def ^:dynamic cuda-device)

(do
  ;;
  ;; Anything backed by a :native-buffer has a zero
  ;; copy pathway to and from numpy.
  ;; Https://clj-python.github.io/libpython-clj/Usage.html
  (alter-var-root #'hd/default-opts
                  (fn [m]
                    (assoc m
                      :tensor-opts {:container-type
                                      :native-heap})))
  (require-python '[numpy :as np])
  (require-python '[torch :as torch])
  (require '[libpython-clj2.python.np-array])
  (alter-var-root
    #'cuda-device
    (constantly
     (if (py.. torch/cuda (is_available)) "cuda" "cpu"))))



(defn ->address-matrix-torch
  [address-length address-count density]
  (py.. (torch/less (torch/rand [address-count
                                 address-length]
                                :device
                                "cuda")
                    density)
    (to torch/float16)))









;; ------------------
;; Literature:
;; ------------------
;;
;; 1
;; Pentti Kanerva 1993 Sparse Distributed Memory and Related Models
;;
;; 2
;; Pentti Kanerva /Sparse Distributed Memory/, 1988
;;
;; 3
;; Jaeckel, L.A. 1989a. An Alternative Design for a Sparse Distributed Memory.
;; Report RIACS TR 89.28, Research Institute for Advanced Computer Science,
;; NASA Ames Research Center.
;;
;; 4
;; Jaeckel, L.A. 1989b. A Class of Designs for a Sparse Distributed Memory. Report
;; RIACS TR 89.30, Research Institute for Advanced Computer Science, NASA
;; Ames Research Center.
