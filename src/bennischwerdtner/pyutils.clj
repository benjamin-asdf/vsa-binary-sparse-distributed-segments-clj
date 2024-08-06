(ns bennischwerdtner.pyutils
  (:require
   [clojure.set :as set]
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

(do
  (require-python '[numpy :as np])
  (require-python '[torch :as torch]))


(def ^:dynamic *torch-device* :cpu)

(alter-var-root
  #'*torch-device*
  (constantly
    (if (py.. torch/cuda (is_available)) :cuda :cpu)))

(defn ensure-torch
  ([tens] (ensure-torch tens *torch-device*))
  ([tens torch-device]
   (cond (dtt/tensor? tens)
         (let [t-numpy (numpy/zeros [(count tens)]
                                    :dtype
                                    numpy/float32)]
           (dtt/tensor-copy! tens
                             (dtt/ensure-tensor t-numpy))
           (torch/tensor t-numpy
                         :dtype torch/float32
                         :device torch-device))
         (= (py/python-type tens) :tensor) tens)))

(defn torch-memory-size
  [t]
  (* (py.. t (element_size)) (py.. t (numel))))

(defn torch->jvm [tens] (py.. tens (to "cpu") (numpy)))
