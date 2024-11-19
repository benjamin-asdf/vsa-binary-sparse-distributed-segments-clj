(ns synapsembles
  (:require
    [tech.v3.datatype.functional :as f]
    [tech.v3.datatype :as dtype]
    [tech.v3.tensor :as dtt]
    [tech.v3.datatype.bitmap :as bitmap]
    [fastmath.random :as fm.rand]
    [fastmath.core :as fm]
    [bennischwerdtner.sdm.sdm :as sdm]
    [bennischwerdtner.hd.binary-sparse-segmented :as hd]
    [bennischwerdtner.pyutils :as pyutils]
    [tech.v3.datatype.unary-pred :as unary-pred]
    [tech.v3.datatype.argops :as dtype-argops]
    [bennischwerdtner.hd.codebook-item-memory :as codebook]
    [bennischwerdtner.hd.ui.audio :as audio]
    [bennischwerdtner.hd.data :as hdd]
    [libpython-clj2.require :refer [require-python]]
    [libpython-clj2.python :refer [py. py..] :as py]))

;; Lit:
;;
;; 1
;; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3005627/
;; Neural syntax: cell assemblies, synapsembles and readers
;; György Buzsáki
;;
;;
;; According to Hebb’s definition (1949), an assembly is characterized by the stronger synaptic connectivity among assembly members than with other neurons. In principle, chains of slow firing neurons, connected with predetermined and fixed synaptic weights can form groups and propagate activity (Abeles, 1991). However, strong, ‘fixed’ connectivity may not be a good model for segregating neuronal groups since synaptic weight distributions are perpetually changing in an activity-dependent fashion in the working brain. In fact, the dynamic range of short-term synaptic plasticity is large and similar to that of long-term plasticity (Marder and Buonomano, 2003), posing problems for the synaptic connection-based definition of cell assemblies. It follows that knowledge of spiking activity is insufficient to properly describe the state of the cortical network unless the distribution of momentary synaptic weights, i.e., the instantaneous functional connection matrix, is also known.


;;
;; 2
;; https://pubmed.ncbi.nlm.nih.gov/19145235/
;; State-dependent computations: spatiotemporal processing in cortical networks
;; Buonomano DV, Maass W. State-dependent computations: spatiotemporal processing in cortical networks. Nat Rev Neurosci. 2009 Feb;10(2):113-25. doi: 10.1038/nrn2558. Epub 2009 Jan 15. PMID: 19145235.
;;


;;

;; ---

;;
;; This is addressed by ensembles.clj via the weight state
;;

;;
;; Here, I want to develop an example synapsembles model that works well together with sdm and block sparse hypervectors.
;;


;; The ensemble:
;;
;; The putative neuronal letter, ensembles which are concatenated in 'gamma wave packets' [1].
;; I model these as address decoders, where the addresses block sparse hypervectors.
;; I.e. it's just a hypervector in this context.
;;
;;

;; --------------

;; It is interesting to note that the time scale of a fast weight is very short
;;
;;






;;
;; Associative memories:
;; -------------------------
;;
;; for this, I use an sdm
;;
;; autoassociative mode    a->a
;;
;; heteroassociative mode  a->b
;;
;;
;; k-fold sequence:        a->b
;;                            b->c
;;                         a---->c
;;
;;

;;
;; Now, I want to extend the vocabulary to synapsemles (temporary 'fast weights')
;;

;;
;; Adding temporary glue-state (synapsembles).
;;
;;
;; Idea 1:
;;
;; Sdm with decaying weights.
;; (can either be *the* sdm, or a separate glue sdm).
;;
;; 1a:
;;
;; - reset the glue sdm after `reset-time` steps.
;;
;; - downside: now globally, temporally evolving 'system level' states
;; - perhaps this downside can be overcome by some kind of nested bridging scheme or something
;;
;;
;; 1b:
;;
;; - decay content matrix bits sdm
;; - either drop or decrement
;; - sounds like the most biologically intuitive, that synapses have some fast strenght that decays,
;;   or synapses have a half time
;;
;; 1c:
;; - same as 1b but drop complete content rows at a time
;;
;;
;;
;;

;; So here is idea 1b:
;; The GlueSpace with a decay function
;;

(defprotocol GlueSpace
  (write [this address content decoder-threshold])
  (lookup [this address top-k decoder-threshold])
  (decay [this drop-chance]))

(defn glue-sdm
  [opts]
  (let [decoder (sdm/->decoder-coo opts)
        storage (sdm/->sdm-storage-coo opts)]
    (reify
      GlueSpace
        (write [this address content decoder-threshold]
          (sdm/write-1 storage
                       (sdm/decode-address
                         decoder
                         address
                         decoder-threshold)
                       content))
        (lookup [this address top-k decoder-threshold]
          (sdm/lookup-1 storage
                        (sdm/decode-address
                          decoder
                          address
                          decoder-threshold)
                        top-k))
        (decay [this drop-chance]
          (sdm/storage-decay storage drop-chance)))))




(comment
  (do
    (def glue (glue-sdm {:address-count 100 :address-density 0.2 :word-length (long 1e4)}))
    (write glue (hdd/clj->vsa* :a) (hdd/clj->vsa* :a) 1)
    (for [n (range 20)]
      (do
        (decay glue 1/4)
        (hd/similarity (hdd/clj->vsa* :a) (pyutils/torch->jvm (:result (lookup glue (hdd/clj->vsa* :a) 1 1)))))))
  '(1.0 1.0 1.0 1.0 1.0 0.9 0.7 0.5 0.3 0.2 0.15 0.1 0.1 0.1 0.05 0.05 0.05 0.05 0.05 0.05)

  ;;
  ;; that last bit is lucky, I feel like that's typical for random decay processes
  ;; (I mean obviously, it has exactly 1/4 chance at each step to decay)
  ;;

  (time
   (do
     (def glue (glue-sdm {:address-count (long 1e6) :address-density 0.00003 :word-length (long 1e4)}))
     (write glue (hdd/clj->vsa* :a) (hdd/clj->vsa* :a) 1)
     ;; (decay glue 1/4)
     (doall
      (for
          [n (range 20)]
          (do
            (decay glue 1/4)
            (hd/similarity (hdd/clj->vsa* :a) (pyutils/torch->jvm (:result (lookup glue (hdd/clj->vsa* :a) 1 1)))))))))
  '(1.0 1.0 1.0 1.0 1.0 0.85 0.65 0.35 0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)

  ;; The dynamics of this is determinend by address-count, address activation probability, the decay rate
  ;; (if the item is stored in more addr. locations, it survives longer)
  ;;
  ;; The ability by the reader to still recognize the hdv has to do with word-length and hdv dennsity I suppose.
  ;;
  ;;

  (time
   (do
     (def glue
       (glue-sdm {:address-count (long 1e6)
                  :address-density 0.00003
                  :word-length (long 1e4)}))
     (write glue (hdd/clj->vsa* :a) (hdd/clj->vsa* :a) 1)
     ;; (decay glue 1/4)
     (doall (for [n (range 20)]
              (do (decay glue 0.1)
                  (hd/similarity
                   (hdd/clj->vsa* :a)
                   (pyutils/torch->jvm
                    (:result (lookup glue
                                     (hdd/clj->vsa* :a)
                                     1
                                     1)))))))))
  '(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0)

  (time (do
          (def glue
            (glue-sdm {:address-count (long 1e6)
                       :address-density 0.00003
                       :word-length (long 1e4)}))
          (write glue (hdd/clj->vsa* :a) (hdd/clj->vsa* :a) 1)
          (doall (for [n (range 20)]
                   (do (decay glue 1/2)
                       (hd/similarity
                        (hdd/clj->vsa* :a)
                        (pyutils/torch->jvm
                         (:result (lookup glue
                                          (hdd/clj->vsa* :a)
                                          1
                                          1)))))))))
  '(1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.95 0.8 0.5 0.3 0.25 0.1 0.05 0.05 0.0 0.0 0.0 0.0 0.0)

  (time
   (do (def glue
         (glue-sdm {:address-count (long 1e6)
                    :address-density 0.00003
                    :word-length (long 1e4)}))
       (doseq [n (range 1000)]
         (write glue (hdd/clj->vsa* n) (hdd/clj->vsa* n) 1))
       (doall (for [n (range 20)]
                (do (decay glue 1/2)
                    (hd/similarity
                     (hdd/clj->vsa* 0)
                     (pyutils/torch->jvm
                      (:result (lookup glue
                                       (hdd/clj->vsa* 0)
                                       1
                                       1)))))))))
  '(1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.95 0.7 0.5 0.3 0.2 0.15 0.1 0.05 0.0 0.0 0.0 0.0 0.0)
  ;; take 1.1s for 100 elements

  ;; 1k takes
  ;; 83.5s
  ;; kinda slow


  )





;; ---------------------------------

























;;
;; Random intrinsic firing rate:
;;
;; Speculative:
;;
;; use pattern 1:
;;
;; create random seed symbols
;; (we can do this the moment we look by making a seed hypervector).
;;
;;
;; use pattern 2:
;;
;; - randomly re-instantiate a 'known' ensemble, i.e. one that has relationships somewhere in the rest of the system.
;; - the speculation is that auto associative cell assemblies have the chance to randomly activate
;;
;; use pattern 3:
;; - Ensemble shift, allowing an active ensemble to recruit neighbouring neurons (from the attractor net)
;;
