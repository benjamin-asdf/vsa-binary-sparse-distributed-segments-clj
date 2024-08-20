;; ---------
;; Notes
;; ---------
(ns analogy-arc.setting-the-tone
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
   [bennischwerdtner.hd.data :as hdd]))




(alter-var-root
 #'hdd/*item-memory*
 (constantly (codebook/codebook-item-memory 1000)))



;; The Problem:
;; ---------------------------
;; - hierachical sequences


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


;; Time is neuronal space in the brain


;; Chunking of messages by agreed rules allows the generation of and reading (i.e., “de-ciphering”) virtually infinite combinations from a finite number of elements in human, sign, body, artificial, and computer languages, music, and mathematical logic, and, presumably, the brain. This small set of rules, usually referred to as syntax, governs the combination and temporal progression of discrete elements (such as letters or musical notes) into ordered and hierarchical relations (words, phrases, and sentences or chords and chord progressions) that allow the generation of messages, which, when interpreted by the receiver, become information. Syntax is exploited in all systems where messages are coded and transmitted, searched for, and decoded.


;; cypher:
;; - marking the message boundaries  by cyclic inhibition partitions a continous stream of events into
;; discrete units and forms athe hypothetical basis for extraction and sythesis of new information
;; (a process called 'abstraction')
;;

;; 1. duty face for sending,
;; 2. pertubation or "receiving" phase


;; - temporal coordination and synchrony have different meanings.
;; - gamma ensembles are seperated with the their negative spike correlations
;; - ~7 gamma ensembles (letters) make word
;;

;;
;;
;;  1 -> 2 -> 3 , ...
;;
;;  This is like hyperdimensional dominos, clack-clack-clack
;;
;; - theta and slower *counter* synchrony
;; - has to do with the this day-night seasons thing I guess
;;
;;
;; - language / syntax might be more than a metaphor
;; - prosodic features are between 0.3Hz and 2Hz - delta band
;; - Syllables 'beat' rhythmically 4 to 8 times per second (theta band)
;; - phonemes and fast transitions 30 and 80 Hz (gamma band)
;; - edges of the sond envelope of speech can reset the ahse of slow brain oscillations
;;









;; -------
;; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3005627/
;;
;; https://www.sciencedirect.com/science/article/pii/S0896627323002143?via%3Dihub




;; (Design WIP)
;; - In order to model Buszaki synapsembles and neuronal syntax:



;; concepts:
;; -----------
;;
;; - Neuronal letter ('gamma assembly'):                      | can be modeled with a hdv (but it's the *readers* that matter)
;; - Synapsembles: 'dynamic glue'                             | My hunch is that a temporary ('resetable') SDM will do the job
;; - Inhibition and Syntax                                    |
;;   Idea is that without syntax and pauses,
;;   activity would run away.
;;


;;
;; 👉 the postulated physiological goal of the cell assembly is to mobilize enough peer neurons so that their collective spiking activity can discharge a target (reader) neurons(s)
;;

;; A discrite, collective unitary event:
;; The 'fundamental cell assembly' or 'assembly τ'
;;
;; I.e. the synchronous (given the integration time of the reader) activity of upstream neurons that activate a reader.
;;
;; This 10-30ms time window (integration time of principal cells) overlaps with
;; - AMPA receptor-mediated EPSPs
;; - GABA-A receptoer-mediated IPSPs
;;
;; - this dynamic makes gamma local field potential LFP
;; - this window corresponds to the window of spike timing-dependent plasticity
;;


;; --
;; 'represntation-based' descriptions:
;; Hebb Cell Assemblies: graph of synatpically interconnected excitatory neurons
;;
;;
;; update to reader-centric definition:
;; --
;; - activiting yourself is useful but not the point
;; - what matters is the reader mechanism
;; - (synfire chain agrees)
;; -> this makes it relational and functional. Otherwise, what is the point of an 'acitve pattern'?
;;    it must have an effect.
;;
;;

;; Quote:
;;
;; I use the term ‘reader’ as a metaphor to refer to a classifier-actuator mechanism. The reader is both an observer-integrator and a decision maker in the sense that it generates a tangible, measurable, and interpretable output. In the simplest case, the output is binary, such as an action potential of a neuron. The reader is not necessarily an independent, isolated unit but it can be part of the assembly itself, much like members of an orchestra, where each member is a reader of others’ actions. Separation of the reader mechanism from the assembly concept is needed only for a disciplined definition of neuronal alliances serving well-defined goals.
;;
;; Basically creating understanding that the claim is not that 'causal alliances' are not relevant.
;; They are, but not at the core of the functionality of assemblies.
;;



;; - 1 ensemble 1 letter
;; -
;; - quote:
;; Therefore, the third hypothesis I advance is that the constituents of the neural syntax are linked together by dynamically changing constellations of synaptic weights (von der Malsburg, 1994), which I refer to as ‘synapsembles’.
;;
;;



;; replay of assemblies in sleep is 10 times faster than at the behavioral time scale
;;
;; - there are no concurrent real time assembly sequences present during slow wave sleep.
;; (suggesting this is required for conscious experience)
;; - neuronal processing is perpetual (also in slow wave sleep)
;;

;; ---
;; dynamic plasticity:
;; - as powerful as long term
;; - activity alone doesn't describe the sate of a network.
;; - activity and short term synaptic weights do
;;
;; - depressing and potentiating synapses have the same fraction.
;; - needed in models for stability
;;
;;

;;
;; 👉 A particular constellatoin of synaptic weights in a definend time window can be conceived of as an assembly of synapses or 'synapsemble'
;;
;; 'communication by synapsembles' is richer than spikes or rates (binary)
;; they encode continous relationships
;;
;;
;; synapsembles are hard to meassure empirically atm.
;;

;; Buzsáki hypothesizes synapsembles may serve a dual role:
;;
;; - they limit hte lifetime of neural words to subsecond to sends time scales
;; - Such self-tuned synapses are likely cirtical in the buid up and termination of assembly activity.
;; . Such self-tuned synapses are likely critical in the build up and termination of assembly activity. This process may be brought about by the depressing excitatory synapses among the active assembly members and/or by potentiated inhibition of the recruited interneurons, assisted by intrinsic neuronal mechanisms, such as firing history-dependence of spike threshold (Henze and Buzsáki, 2001).
;;
;; - second, synapslembles link neurnal words seperated by cessation of spiking activyt
;; - depresising the inhbitory connections and/or ptentiating excitatory synapses between members of the reding and trailing cell assemblies
;;   may achive such linking
;;

;; -------------------
;; interneurons:
;; segrate assemblies into functional gruops
;;
;; - interneuron-guided grouping - formation of a candidate assembly
;; - exictatory neurons can enslave interneurons in different constelllations
;; -
;; - all of synapsembles, active assembley, interneurons and silenced population are in an exquisite relationship
;; -
;;

;;
;; in neocortex, inhibition can have either a positive or inverse correlation with excitattory thalamic inputi
;; shape the response to On and Off transitions of the stimulus
;; or affect hte tuning properties of principal cells
;;

;; I stop taking note of everything now

;; ---

;; They are part of an assembly and also suppress competing assemblies


;; In sparse network:
;; only a few principal cells discharge an interneuron
;; the interneuron copies the principal cell's firing pattern
;; in turn the transientally interneuron   can suppress the activity of competing principal cells in the vinicitty of their mostly local
;; axon collaterral.
;; As a result, only asingle assembly, (the 'winner') may be active aat a atime evennn in a large neuronal volume.
;;



;; ---
;; Cell Assembly Size
;;
;; 1% of hippocampal pyramidal cells fire ina 20msec time window
;; during theta-related behaviours
;; 15k to 30k CA3 pyr cells converge on a CA1 pyramidal neuron
;; on avr. 150 to 300 CA3 pyramidal cells firing within a gamma cycle comprise an assembly
;;
;; under special conditions,
;; when the inputs converge ont e same dendritic branch and fire synch. in <6 msec, as few as 20 neurons may be sufficient to initiate a forward-propagating dedritic spike
;; -maybe during sharp wave ripples, and in geniculo-cortical system during visual transmission
;;
;;

;;;; ----------
;; small number of neurons gives most of the readout power
;; - same for visual, motor and CA3 readout
;; - unkown, but generally you have large assemblies. At the same time a small fraction can make a good enough signal
;;
;;

;; ----------------
;; readers:
;;
;; - different readers might read different aspects like frequency
;; - Question: how to segregate complex trajectories (i.e. sequence comes in, how do you know the order and members)
;;
;;

;; ---------
;; reader initiation:
;; - not a passive ever ready reciever,
;; Instead, the reader plays the initiating role by temporally biasing activity in the source networks and creating time windows within which the reader can most effectively receive information (Figure 10; Sirota et al., 2003; 2008)
;; )
;; --
;;
;;
;; Each sensory system has co-evolved with such a reader-initiated transfer mechanism. Dedicated motor outputs, such as saccadic eye movements, licking, sniffing, whisking, touching, twitching of the inner ear muscles or other gating mechanisms assist their specific sensory systems by ‘resetting’ or synchronizing spiking activity in large parts of the corresponding sensory system and/or creating transient gains, which enhance the reader (sensory) system’s ability to process the inputs (Ahissar and Arieli, 2001; Bremmer et al., 2009; Desimone and Ungerleider, 1989; Guiterrez et al., 2010; Halpern, 1983; Henson 1965; Kepecs et al., 2006; Kleinfeld et al., 2006).
;;)

;; -> the sensors are used by the reader
;;
;; - the reader sends an output command to optimize the sensor
;; - synchronizing-blanking mechanisms
;; - they generate transient gains
;;

;;
;;
;; same tings for the inner parts of the brain
;;
;; slow wave sleep: neocortex (reader) reads from hippocampus
;;
;; waking brain: other way around
;;






;; --------------------
;; constantly world
;; 'identity' world
;;

(def constantly-world (constantly :a))

(def flip-flop-world
  {[:a :right] :b
   [:b :left] :a})

(def triangle-world
  {[:a :right] :b

   [:b :right] :c
   [:b :left] :a

   [:c :left] :b})

;; triangle world is the superposition of flip flop worlds

(structurally
 [flip-flop-world {:a :a :b :b}]
 [flip-flop-world {:a :b :b :c}])


(hdd/clj->vsa*
 [:+
  [:*> flip-flop-world]
  [:*> flip-flop-world]])


(hdd/cleanup
 (hdd/clj->vsa*
  [:.
   [:+
    [:* :a :right :b]
    [:* :b :left :a]]
   [:* :b :left]]))

;; -----------------------------------------------
;; memory

(defprotocol GlueSpace
  (put [this addr content])
  (retrieve [this addr top-k]))

(defn ->gluespace
  []
  (let [sdm (sdm/->sdm {:address-count (long 1e6)
                        :address-density 0.000003
                        :word-length (long 1e4)})]
    (reify
      GlueSpace
        (put [this addr content]
          (sdm/write sdm addr content 1))
        (retrieve [this addr-prime top-k]
          (let [lookup-outcome
                  (sdm/lookup sdm addr-prime top-k 1)]
            (when (< 0.1 (:confidence lookup-outcome))
              (some-> lookup-outcome
                      :result
                      sdm/torch->jvm
                      (dtt/->tensor :datatype :int8))))))))



;; an empty gluespace is like 2mb
;; So 1000 empty ones is 2Gb
;;

(def gluespace (->gluespace))

;; -------------------------------------------------
;; analogy structure
;;


(defprotocol Mapping
  (translate [this source])
  (superposition [this & other]))

(defprotocol World
  (move [this action state])
  (superposition [this & other])
  (tag [this])
  ;; (bind [h])
  )

(defprotocol HyperAtom
  (halo [this])
  (core [this])
  (overlap [this & other])
  (difference [this & other]))

(defprotocol HyperWord
  (children [this]))

(hdd/cleanup
 (hdd/clj->vsa*
  [:.
   [:+
    [:* :a :right :b]
    [:* :b :left :a]]
   [:* :b :left]]))


;; ----------------------------------------------
;; Trajectories
;;

(defprotocol Trajectory)

(defprotocol TrajectoryEngine
  (bind [this trajectory terminal])
  (seed
    ;; new trajectory
    [this])
  ;; ? query?
  (replay [this query]))


;; -----------------------------------------------

(defn ->glue-mapping
  [mapping]
  (let [glue (->gluespace)]
    (doseq [[k v] mapping]
      (let [k (hdd/clj->vsa* k)
            v (hdd/clj->vsa* v)]
        (do (put glue k v) (put glue v k))))
    (fn [source] (retrieve glue (hdd/clj->vsa* source) 1))))

(comment
  (hdd/cleanup
   (translate (->glue-mapping {:b :a :c :b})
              (hdd/clj->vsa* :b))))

(defn ->world
  [finite-state-automaton]
  (let [glue (->gluespace)
        _ (doseq [[[state action] destination]
                  finite-state-automaton]
            (put glue
                 (hdd/clj->vsa* [:* state action])
                 (hdd/clj->vsa* destination)))
        ]
    (reify
      World
      (move [this action state]
        (retrieve glue
                  (hdd/clj->vsa* [:* state action])
                  1))
      )))

(defn structurally
  [worlds-mappings]
  (for [[world mapping] worlds-mappings])
  (reify
    World
    (move [this action state] (hdd/clj->vsa* [:+]))))

(let [mapping
      ;; (hdd/clj->vsa*
      ;;  {:b :a :c :b})
      (->glue-mapping {:b :a :c :b})
      source-word (hdd/clj->vsa* [:+ [:* :a :right :b]
                                  [:* :b :left :a]])
      ana (reify
            World
            (move [this action state]
              (mapping (hdd/clj->vsa*
                        [:. source-word action
                         (mapping state)]))))]
  (hdd/cleanup (move ana :left :c)))

(let [mapping (->glue-mapping {:b :a :c :b})
      source-world (->world flip-flop-world)
      ana (reify
            World
            (move [this action state]
              (mapping (move source-world
                             action
                             (mapping state)))))]
  (hdd/cleanup (move ana :left :c)))

(put gluespace (hdd/clj->vsa* [:* :a :s]) (hdd/clj->vsa* :c))
(put gluespace (hdd/clj->vsa* [:* :a2 :s2]) (hdd/clj->vsa* :c2))

(hdd/cleanup*
 (retrieve
  gluespace
  (hdd/clj->vsa* [:+ [:* :a2 :s2] [:* :a :s]])
  2))
'(:c :c2)

(hdd/cleanup*
 (retrieve
  gluespace
  (hdd/clj->vsa* [:-- [:+ [:* :a2 :s2] [:* :a :s]] 0.5])
  2))
'(:c :c2)

(hdd/cleanup*
 (retrieve
  gluespace
  (hdd/clj->vsa* [:-- [:* :a2 :s2] 0.5])
  2))
'(:c2)


;; superimpose worlds



;;
;; - a world has tag
;; - a mapping is a bind between 2 worlds
;; - worlds can be incomplete,
;;

(let [w1 (hdd/clj->vsa* :w1)
      w2 (hdd/clj->vsa* :w2)]
  ;; separate worlds
  ;;
  (put
   gluespace
   (hdd/clj->vsa* [:* :a :s w1])
   (hdd/clj->vsa* [:* :c w1]))
  (put
   gluespace
   (hdd/clj->vsa* [:* :a2 :s2 w2])
   (hdd/clj->vsa* [:* :c2 w2])))

(let
    [w1 (hdd/clj->vsa* :w1)
     w2 (hdd/clj->vsa* :w2)
     mapping (hd/bind w1 w2)]
    (hdd/cleanup*
     (hdd/clj->vsa*
      [:.
       (retrieve gluespace
                 (hdd/clj->vsa*
                  [:* :a :s [:. mapping w2]])
                 1)
       [:. mapping w2]])))


(defn ->trajectory []
  (dtt/->tensor (repeatedly 7 hd/->hv) :datatype :int8))

;; (defn ->trajectory-engine
;;   []
;;   (let [k-fold-memory (sdm/->k-fold-memory
;;                         {:address-count (long 1e6)
;;                          :address-density 0.000007
;;                          :k-delays 3
;;                          :stop? (fn [acc next-outcome]
;;                                   (when (< (:confidence
;;                                              next-outcome)
;;                                            0.05)
;;                                     :low-confidence))
;;                          :word-length (long 1e4)})]
;;     (reify
;;       TrajectoryEngine
;;         (seed [this]
;;           (let [trajectory (repeatedly 7 hd/->seed)]
;;             (sdm/reset k-fold-memory)
;;             (sdm/write-xs! k-fold-memory trajectory 1)
;;             trajectory))
;;         (replay [this q]
;;           (sdm/reset k-fold-memory)
;;           (->> (sdm/lookup-xs k-fold-memory q 1)
;;                :result-xs
;;                (map :result)
;;                (map pyutils/ensure-jvm))))))

(defn ->trajectory-engine
  []
  (let [sdm (sdm/k-fold-sdm {:address-count (long 1e6)
                             :address-density 0.000006
                             :k-delays 5
                             :word-length (long 1e4)})]
    (reify
      TrajectoryEngine
        (seed [this]
          (let [trajectory (repeatedly 7 hd/->seed)]
            (sdm/reset sdm)
            (doseq [[addr content]
                      (partition 2 1 trajectory)]
              (sdm/write sdm addr content 1))
            trajectory))
        (replay [this q]
          (sdm/reset sdm)
          (->> (sdm/converged-lookup-impl
                 sdm
                 (hdd/clj->vsa* q)
                 {:decoder-threshold 1
                  :stop? (fn [acc next-outcome]
                           (when (< (:confidence
                                      next-outcome)
                                    0.05)
                             :low-confidence))
                  :top-k 1})
               :result-xs
               (map :result)
               (map pyutils/ensure-jvm))))))


;; ---------------------

(defn ->audio [{:keys [frequency duration]}]
  {:frequency frequency
   :duration duration})

(defn play!
  [audio]
  (clojure.java.shell/sh
    "ffplay"
    "-nodisp"
    "-f"
    "lavfi"
    "-i"
    (str "sine=frequency=" (:frequency audio)
         ":duration=" (:duration audio))
    "-autoexit"))

(play! (->audio {:frequency 440 :duration 0.2}))
(play! (->audio {:frequency 1200 :duration 0.2}))
(play! (->audio {:frequency 80 :duration 0.2}))

(defn render-hd-frequency
  [hd]
  (+ 80
     (*
      (- 1100 80)
      ;; between 0 and 500
      (/ (first (hd/hv->indices hd)) 500))))

(defn render-hd-duration
  [hd]
  (+
   0.05
   (* 0.3 (/ (first (hd/hv->indices hd)) 500))))

(defn render-hd-audio
  [hd]
  (->audio {:frequency (render-hd-frequency hd)
            :duration
              ;; (render-hd-duration hd)
              0.05}))
(play!
 (->audio {:frequency (render-hd-frequency (hd/->seed)) :duration 0.1}))

(defn listen! [hdvs]
  (future (doseq [hdv hdvs]
            (play! (render-hd-audio hdv)))))

(def trajector (->trajectory-engine))

(def x (seed trajector))

(def t1 [(seed trajector) (seed trajector) (seed trajector)])
(def t2 [(seed trajector) (seed trajector) (seed trajector)])
(def t3 [(seed trajector) (seed trajector) (seed trajector)])
(def t4 [(seed trajector) (seed trajector) (seed trajector)])

(defn listen-seqs!
  [seq]
  (doseq [x seq]
    (listen! x)
    (Thread/sleep (rand-nth [0 25 50]))))

(defn listen-seq!
  [seq]
  (doseq [x seq]
    (listen! (replay trajector (first x)))
    (Thread/sleep (rand-nth [50 100]))))


(listen-seq! t1)
(listen-seq! t2)
(listen-seq! t3)
(listen-seq! t4)

(listen-seq!
 [(seed trajector) (seed trajector) (seed trajector)])


(def alphabet (into [] (map (comp keyword str char) (range (int \a) (inc (int \z))))))

(listen! (map hdd/clj->vsa* alphabet))

;; abc
(listen! (take 3 (map hdd/clj->vsa* alphabet)))
(def abc-t (seed trajector))

(listen-seqs!
 [(take 3 (replay trajector (first abc-t)))
  (take 3 (map hdd/clj->vsa* alphabet))])

(map
 (fn [a b] (hd/similarity a b))
 abc-t
 (replay trajector (first abc-t)))


(listen-seqs!
 [(let [focus (into [] (replay trajector (first abc-t)))
        sensoric-focus
        (into [] (take 3 (map hdd/clj->vsa* alphabet)))]
    (for [idx (range 7)]
      (if-let [sf (get sensoric-focus idx)]
        (hd/bind (nth focus idx) sf)
        (nth focus idx))))])

(listen-seqs! [(take 3 (replay trajector (first abc-t)))])
(listen-seqs! [(take 3 (map hdd/clj->vsa* alphabet))])
(listen-seqs! [(take 7 (map hdd/clj->vsa* alphabet))])

(listen-seqs!
 [(replay trajector (first abc-t))
  (take 7 (map hdd/clj->vsa* alphabet))
  (map hd/bind
       (replay trajector (first abc-t))
       (take 7 (map hdd/clj->vsa* alphabet)))])

(listen-seqs!
  (let [focus (into [] (replay trajector (first abc-t)))
        sensoric-focus
          (into [] (take 3 (map hdd/clj->vsa* alphabet)))]
    [sensoric-focus
     focus
     (map hd/bind focus sensoric-focus)]))

(let [a (take 3 (seed trajector))
      b (take 3 (seed trajector))
      c (take 3 (seed trajector))
      abc (map hdd/clj->vsa* [:a :b :c])]
  (listen-seqs!
    [(concat
      abc
      (map hd/bind a (repeat 3 (first abc)))
      abc
      (map hd/bind b (repeat 3 (second abc)))
      abc
      (map hd/bind c (repeat 3 (nth abc 2))))]))













(comment
  (time
   (let [k-fold-memory (sdm/->k-fold-memory
                        {:address-count (long 1e5)
                         :address-density 0.0026
                         :k-delays 3
                         :stop? (fn [acc next-outcome]
                                  (when (< (:confidence
                                            next-outcome)
                                           0.05)
                                    :low-confidence))
                         :word-length (long 1e4)})]
     (doseq [n (range 100)]
       (let [trajectory (repeatedly 5 hd/->seed)]
         (time (sdm/write-xs! k-fold-memory trajectory 2))
         (sdm/reset k-fold-memory)))
     (let [trajectory (map hdd/clj->vsa* [:a :b :c :d :e])]
       (sdm/write-xs! k-fold-memory trajectory 2)
       (sdm/reset k-fold-memory)
       (->>
        (sdm/lookup-xs k-fold-memory (hdd/clj->vsa* :a) 2)
        :result-xs
        (map :result)
        (map pyutils/ensure-jvm)
        (map hdd/cleanup*)))))

  '((:a) (:b) (:c) (:d) (:e))

  (time
   (let [k-fold-memory (sdm/->k-fold-memory
                        {:address-count (long 1e6)
                         :address-density 0.000007
                         :k-delays 3
                         :stop? (fn [acc next-outcome]
                                  (when (< (:confidence
                                            next-outcome)
                                           0.05)
                                    :low-confidence))
                         :word-length (long 1e4)})]
     (doseq [n (range 5)]
       (let [trajectory (repeatedly 5 hd/->seed)]
         (time (sdm/write-xs! k-fold-memory trajectory 1))
         (sdm/reset k-fold-memory)))


     (let [trajectory (map hdd/clj->vsa* [:a :b :c :d :e])]
       (sdm/write-xs! k-fold-memory trajectory 1)
       (sdm/reset k-fold-memory)
       (->>
        (sdm/lookup-xs k-fold-memory (hdd/clj->vsa* :a) 1)
        :result-xs
        (map :result)
        (map pyutils/ensure-jvm)
        (map hdd/cleanup*)))))

  )




(comment
  (def m
    (sdm/k-fold-sdm {:address-count (long 1e6)
                     :address-density 0.000006
                     :k-delays 5
                     :word-length (long 1e4)}))
  (sdm/write m
             (hdd/clj->vsa* (first alphabet))
             (hdd/clj->vsa* (second alphabet))
             1)
  (sdm/write m
             (hdd/clj->vsa* (nth alphabet 1))
             (hdd/clj->vsa* (nth alphabet 2))
             1)
  (do (sdm/reset m)
      (->
       (sdm/lookup m (hdd/clj->vsa* (first alphabet)) 1 1)
       :result
       pyutils/torch->jvm
       hdd/cleanup*))
  (do (sdm/reset m)
      (let [out (:result (sdm/lookup m
                                     (hdd/clj->vsa*
                                      (first alphabet))
                                     1
                                     1))]
        (-> (sdm/lookup m out 1 1)
            :result
            pyutils/torch->jvm
            hdd/cleanup*)))
  (doseq [[addr content] (partition 2 1 alphabet)]
    (sdm/write m
               (hdd/clj->vsa* addr)
               (hdd/clj->vsa* content)
               1))
  (map (fn [a b] (hd/similarity a b))
       (map hdd/clj->vsa* alphabet)
       (do (sdm/reset m)
           (->> (sdm/converged-lookup-impl
                 m
                 (hdd/clj->vsa* (first alphabet))
                 {:decoder-threshold 1
                  :stop? (fn [acc next-outcome] false)
                  :top-k 1})
                :result-xs
                (map :result)
                (map pyutils/ensure-jvm)
                ;; (map hdd/cleanup*)
                )))
  (f/sum (second *1))
  '((:a) (:b) (:c) (:d) (:e) (:f) (:g) (:h)))




















;;
;; gluespace as neuronal area
;; ---------------------------------
;; - sdm is eq. to Hopfield net
;; - my feel is an sdm might be a good enough universal neuronal net component
;; - it's more powerful and versatile by splitting update and query
;;




;; factorization

(comment

  (def actions [:left :right])
  (def a (hdd/clj->vsa* [:* :right :left :right]))

  (put gluespace (hdd/clj->vsa* :left) (hdd/clj->vsa* :left))
  (put gluespace (hdd/clj->vsa* :right) (hdd/clj->vsa* :right))
  (retrieve gluespace (hdd/clj->vsa* (into #{} actions)) (count actions))

  (defn randomize-drop
    [hv drop-chance]
    (hd/indices->hv*
     (for [idx (hd/hv->indices hv)]
       (when (zero? (fm.rand/flip drop-chance)) [idx]))))

  (hd/indices->hv* [[]])
  (f/sum (randomize-drop a 0.9))

  (map hdd/cleanup*
       (let [p (hdd/clj->vsa* (into #{} actions))
             x (hd/unbind p
                          (retrieve gluespace
                                    (randomize-drop
                                     (hdd/clj->vsa*
                                      (into #{} actions))
                                     (/ (dec (count actions))
                                        (count actions)))
                                    1))
             y (hd/unbind p
                          (randomize-drop
                           (hdd/clj->vsa* (into #{} actions))
                           (/ (dec (count actions))
                              (count actions))))
             z (hd/unbind p
                          (randomize-drop
                           (hdd/clj->vsa* (into #{} actions))
                           (/ (dec (count actions))
                              (count actions))))]
         [(hd/unbind p (hd/bind y z))
          (hd/unbind p (hd/bind x z))
          (hd/unbind p (hd/bind x y))]))




  ;; this seems to settle after 2 iterations:

  (for [n (range 100)]
    (= (into
        #{}
        (mapcat identity

                (let
                    [p (hdd/clj->vsa* [:* :right :left])]
                    (map hdd/cleanup*
                         (let [x (retrieve gluespace
                                           (randomize-drop
                                            (hdd/clj->vsa*
                                             (into #{} actions))
                                            (/ (dec (count actions))
                                               (count actions)))
                                           1)
                               y (retrieve gluespace
                                           (randomize-drop
                                            (hdd/clj->vsa*
                                             (into #{} actions))
                                            (/ (dec (count actions))
                                               (count actions)))
                                           1)]
                           (let [x (hd/unbind p y)
                                 y (hd/unbind p x)
                                 x (retrieve gluespace
                                             (randomize-drop
                                              x
                                              (/ (dec (count actions))
                                                 (count actions)))
                                             1)
                                 y (retrieve gluespace
                                             (randomize-drop
                                              y
                                              (/ (dec (count actions))
                                                 (count actions)))
                                             1)
                                 x (hd/unbind p y)
                                 y (hd/unbind p x)]
                             [x y]))))


                ))
       #{:right :left}))


  ;; sdm resonator:
  ;; -------------------

  (+ 1)



  (defn resonate
  [target depth actions]
  (let [gluespace (->gluespace)
        _ (doseq [action actions]
            (put gluespace
                 (hdd/clj->vsa* action)
                 (hdd/clj->vsa* action)))]
    (reductions
      (fn [{:keys [estimates]} n]
        (audio/listen! estimates)
        (let [estimates
                (into []
                      ;; random query in superposition
                      (comp
                        (map (fn [x]
                               (randomize-drop
                                 x
                                 (/ (dec (count actions))
                                    (count actions)))))
                        (map (fn [x]
                               (retrieve gluespace x 1))))
                      estimates)]
          (def estimates estimates)
          (def target target)
          {:estimates
             (into []
                   (for [idx (range (count estimates))]
                     (hd/unbind
                       target
                       ;; ---
                       (hd/bind*
                         (for [j (range (count estimates))
                               :when (not= j idx)]
                           (nth estimates j))))))
           :n n}))
      {:estimates (into []
                        (repeat depth
                                (hdd/clj->vsa*
                                  (into #{} actions))))}
      (range 3))))


  (map hdd/cleanup*
       (:estimates
        (last (take 3
                    (resonate (hdd/clj->vsa*
                               [:* :right :left])
                              2
                              [:left :right])))))






  (map
   hdd/cleanup*
   (:estimates (last (take 2
                           (resonate
                            (hdd/clj->vsa* [:* :right :left :right])
                            2
                            [:left :right]))))))





(defn resonate
  [target depth actions]
  (let [gluespace (->gluespace)
        ;; _ (doseq [n (range depth)]
        ;;     (doseq [action actions]
        ;;       (put gluespace
        ;;            (hdd/clj->vsa* [:> action n])
        ;;            (hdd/clj->vsa* [:> action n]))))
       ]
    (put gluespace
         (hdd/clj->vsa* [:> :left])
         (hdd/clj->vsa* [:> :left]))
    (put gluespace
         (hdd/clj->vsa* [:> :right])
         (hdd/clj->vsa* [:> :right]))
    (put gluespace
         (hdd/clj->vsa* :right)
         (hdd/clj->vsa* :right))
    (put gluespace
         (hdd/clj->vsa* :left)
         (hdd/clj->vsa* :left))
    (reductions
      (fn [estimates n]
        (audio/listen! estimates)
        (let [estimates
                (into []
                      ;; random query in superposition
                      (comp
                        (map
                         (fn [x]
                           (randomize-drop
                            x
                            0.5
                            ;; (/ (dec (count actions))
                            ;;    (count actions))
                            )))
                        (map (fn [x]
                               (retrieve gluespace x 1))))
                      estimates)]
          (def estimates estimates)
          (def target target)
          (into []
                (for [idx (range (count estimates))]
                  (hd/unbind target
                             ;; ---
                             (hd/bind*
                               (for [j (range (count
                                                estimates))
                                     :when (not= j idx)]
                                 (nth estimates j))))))))
      [(hdd/clj->vsa* #{(hdd/clj->vsa* :left)
                        (hdd/clj->vsa* :right)})
       (hdd/clj->vsa* #{(hdd/clj->vsa* [:> :left])
                        (hdd/clj->vsa* [:> :right])})]
      (range 3))))


(map hdd/cleanup*
     (last (take 3
                 (resonate (hdd/clj->vsa* [:*> :right :left])
                           2
                           [:left :right]))))

;; ---------------------------


(def p (hdd/clj->vsa* [:* :left :left]))

(retrieve gluespace (hd/->empty) 1)

(for [n (range 100)]
  (=
    [:left :left]
    (into
      []
      (map hdd/cleanup
        (let [x (retrieve gluespace
                          (randomize-drop
                            (hdd/clj->vsa* (into #{}
                                                 actions))
                            (/ (dec (count actions))
                               (count actions)))
                          1)
              y (retrieve gluespace
                          (randomize-drop
                            (hdd/clj->vsa* (into #{}
                                                 actions))
                            (/ (dec (count actions))
                               (count actions)))
                          1)]
          (let [[x y]
                  (loop [x (hd/unbind p y)
                         y (hd/unbind p x)
                         n 0]
                    (if (= n 5)
                      [x y]
                      (let [x (retrieve gluespace
                                        x
                                        1
                                        ;; (if (odd? n) 2
                                        ;; 1)
                              )
                            y (retrieve gluespace
                                        y
                                        1
                                        ;; (if (odd? n) 2
                                        ;; 1)
                              )
                            _ (def x x)
                            _ (def y y)
                            x (if (and x (not y))
                                (hdd/difference
                                  (hdd/clj->vsa*
                                    (into #{} actions))
                                  x)
                                x)
                            y (if (and y (not x))
                                (hdd/difference
                                  (hdd/clj->vsa*
                                    (into #{} actions))
                                  y)
                                y)
                            x (or x
                                  (randomize-drop
                                    (hdd/clj->vsa*
                                      (into #{} actions))
                                    (/ (dec (count actions))
                                       (count actions))))
                            y (or y
                                  (randomize-drop
                                    (hdd/clj->vsa*
                                      (into #{} actions))
                                    (/ (dec (count actions))
                                       (count actions))))]
                        (recur (hd/unbind p y)
                               (hd/unbind p x)
                               (inc n))))
                    ;; [x y]
                  )]
            ;; [x y (hd/similarity p (hd/bind x y))
            ;; (map hdd/cleanup* [x y (hd/unbind p y)
            ;; (hd/unbind p x)])]
            [x y]))))))



(defn resonate
  [actions gluespace target depth]
  (loop [estimates (into []
                         (for [d (range depth)]
                           (randomize-drop
                             (hd/permute-n (hdd/clj->vsa*
                                             (into #{}
                                                   actions))
                                           d)
                             (/ (dec (count actions))
                                (count actions)))))
         n 0]
    (if (= n 5)
      estimates
      (let [fail? (some nil? estimates)
            estimates (if-not fail?
                        estimates
                        (into
                          []
                          (map-indexed
                            (fn [idx x]
                              (if x
                                ;; (hd/permute-n
                                ;;  (hdd/clj->vsa*
                                ;;   (into #{} actions))
                                ;;  idx)
                                ;; then x is
                                ;; contributing to a
                                ;; fail, restart and
                                ;; try wihout x

                                (hdd/difference
                                 (hd/permute-n
                                  (hdd/clj->vsa*
                                   (into #{} actions))
                                  idx)
                                 (randomize-drop x 0.5))

                                (randomize-drop
                                 (hd/permute-n
                                  (hdd/clj->vsa*
                                   (into #{} actions))
                                  idx)
                                 (/ (dec (count actions))
                                    (count actions)))))
                            estimates)))]
        (recur (into []
                     (map (fn [x] (retrieve gluespace x 1)))
                     (for [idx (range (count estimates))]
                       (hd/unbind
                         target
                         (hd/bind*
                           (for [j (range (count estimates))
                                 :when (not= j idx)]
                             (nth estimates j))))))
               (inc n))))))


(def depth 5)

(doseq [action actions n (range depth)]
  (put gluespace (hdd/clj->vsa* [:> action n]) (hdd/clj->vsa* [:> action n])))


(for [n (range 100)]
  (map (fn [x] (when x (hdd/cleanup x)))
       (map-indexed
        (fn [idx x] (when x (hd/permute-n x (- idx))))
        (resonate [:left :right]
                  gluespace
                  (hdd/clj->vsa* [:*> :right :left])
                  2))))





(let [x (retrieve gluespace
                  (randomize-drop (hdd/clj->vsa*
                                   (into #{} actions))
                                  (/ (dec (count actions))
                                     (count actions)))
                  1)
      y (retrieve gluespace
                  (randomize-drop (hdd/clj->vsa*
                                   (into #{} actions))
                                  (/ (dec (count actions))
                                     (count actions)))
                  1)]
  (let [[x y]
        (loop [x (hd/unbind p y)
               y (hd/unbind p x)
               n 0]
          (if (= n 5)
            [x y]
            (let [x (retrieve gluespace
                              x
                              1
                              ;; (if (odd? n) 2 1)
                              )
                  y (retrieve gluespace
                              y
                              1
                              ;; (if (odd? n) 2 1)
                              )
                  _ (def x x)
                  _ (def y y)
                  x (if (and x (not y))
                      (hdd/difference
                       (hdd/clj->vsa* (into #{} actions))
                       x)
                      x)
                  y (if (and y (not x))
                      (hdd/difference
                       (hdd/clj->vsa* (into #{} actions))
                       y)
                      y)
                  x (or x
                        (randomize-drop
                         (hdd/clj->vsa* (into #{}
                                              actions))
                         (/ (dec (count actions))
                            (count actions))))
                  y (or y
                        (randomize-drop
                         (hdd/clj->vsa* (into #{}
                                              actions))
                         (/ (dec (count actions))
                            (count actions))))]
              (recur (hd/unbind p y)
                     (hd/unbind p x)
                     (inc n))))
          ;; [x y]
          )]
    ;; (map hdd/cleanup* [x y])
    [ ;; x y
     (hd/similarity p (hd/bind x y))
     (map hdd/cleanup*
          [x y (hd/unbind p y) (hd/unbind p x)])]))




































;; a k-fold sdm is kinda an alternative bind
;;

(defn ->trajectory-engine
  [{:keys [k-delays trajectory-length]}]
  (let [sdm (sdm/k-fold-sdm {:address-count (long 1e6)
                             :address-density 0.000006
                             :k-delays k-delays
                             :word-length (long 1e4)})]
    (reify
      TrajectoryEngine
      (seed [this]
        (let [trajectory (repeatedly trajectory-length
                                     hd/->seed)]
          (sdm/reset sdm)
          (doseq [[addr content]
                  (partition 2 1 trajectory)]
            (sdm/write sdm addr content 1))
          trajectory))
      (bind [this trajectory terminal]
        (sdm/reset sdm)
        (doseq [[addr content] (partition
                                2
                                1
                                (reverse
                                 (concat [terminal] trajectory)))]
          (sdm/write sdm addr content 1)))
      (replay [this q]
        (sdm/reset sdm)
        (->> (sdm/converged-lookup-impl
              sdm
              (hdd/clj->vsa* q)
              {:decoder-threshold 1
               :stop? (fn [acc next-outcome]
                        (when (< (:confidence
                                  next-outcome)
                                 0.05)
                          :low-confidence))
               :top-k 1})
             :result-xs
             (map :result)
             (map pyutils/ensure-jvm))))))

(def trajector
  (->trajectory-engine {:k-delays 2 :trajectory-length 3}))

(def x (seed trajector))


;; resonate by reading from both ends

(let [target (hdd/clj->vsa* :a)
      _ (bind trajector x target)]
  ;; [(replay trajector :a)
  ;;  (replay trajector (hdd/clj->vsa* (into #{} x)))]
  )

(audio/listen! (replay trajector :a))
(hdd/cleanup* (first (replay trajector :a)))

(map
 #(hd/similarity %1 %2) x
 (reverse (drop 1 (replay trajector :a))))

(map #(hd/similarity %1 %2) (reverse x) (replay trajector (last x)))

(map #(hd/similarity %1 %2) (reverse x) (drop 1 (replay trajector :a)))


(let
    [[backwards forwards]
     [
      (replay trajector :a)
      ;; (replay trajector (hdd/clj->vsa* (into #{} x)))
      (replay trajector (hdd/clj->vsa* (first x)))]]

    (audio/listen! (reverse backwards))
    (audio/listen! forwards)

    ;; (map #(hd/superposition %1 %2) (reverse backwards) forwards)

    (map
     #(hd/similarity %1 %2)
     (drop 1 (reverse backwards))
     forwards)

    )



;; -----------------------------
;; The resonator SDM
;;

;;
;; global: hdv
;; for each bit: there is a local sdm
;;
