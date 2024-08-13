(ns bennischwerdtner.hd.hdc
  (:require [tech.v3.tensor :as dtt]
            [bennischwerdtner.hd.binary-sparse-segmented :as
             hd]))

(defn preallocated-alphabet
  [n]
  (dtt/->tensor (repeatedly n hd/->seed) :datatype :int8))
