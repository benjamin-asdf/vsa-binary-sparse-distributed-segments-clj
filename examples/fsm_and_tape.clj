(ns fsm-and-tape
  (:require
    [bennischwerdtner.sdm.sdm :as sdm]
    [tech.v3.datatype.functional :as f]
    [tech.v3.datatype :as dtype]
    [tech.v3.tensor :as dtt]
    [tech.v3.datatype.bitmap :as bitmap]
    [fastmath.random :as fm.rand]
    [bennischwerdtner.hd.binary-sparse-segmented :as hd]
    [bennischwerdtner.pyutils :as pyutils]
    [tech.v3.datatype.unary-pred :as unary-pred]
    [tech.v3.datatype.argops :as dtype-argops]
    [bennischwerdtner.hd.prot :as prot]
    [bennischwerdtner.hd.codebook-item-memory :as codebook]
    [bennischwerdtner.hd.ui.audio :as audio]
    [bennischwerdtner.hd.data :as hdd]))

(alter-var-root
 #'hdd/*item-memory*
 (constantly (codebook/codebook-item-memory 1000)))

(defn read-1
  [sdm addr top-k]
  (let [lookup-outcome (sdm/lookup sdm addr top-k 1)]
    (def lookup-outcome lookup-outcome)

    (when (< 0.2 (:confidence lookup-outcome))
      (some-> lookup-outcome
              :result
              sdm/torch->jvm
              (dtt/->tensor :datatype :int8)))))

;; Turing machine:
;; Finite State Machine and a Tape
;;

;; FSM, is a list of transition tuples:
;;
;; [ old-state, input-symbol, new-state, output-symbol, action ]


;; tape:
;;
;; read-tape
;; write, grows left and right
;;


(defprotocol Tape
  (write [this s])
  ;; move and read have a top-k argument, specifying the degree (count)
  ;; of superposition (roughly in units of seed vectors worth of bits)
  ;;
  ;;
  (read-tape [this]
    [this top-k])
  (move
    [this direction]
    [this direction top-k]))

(defprotocol FSM
  (transition [this current-state input-symbol]))

;; This will require some philosophical explanation,
;; but I think it makes sense.
;; implemented via vsa bind/unbind.
(defprotocol Entanglement
  (flip [this input]))

(defn entangle [symbols]
  (hd/bind* (map hdd/clj->vsa* symbols)))

(extend-protocol Entanglement
  Object
  (flip [this other]
    (hdd/clj->vsa* [:. this other])))

;; ----

(defn sdm-fsm-target
  [target-state output action]
  {:action action
   :output output
   :target-state target-state})

(defn sdm-outcome->action [outcome] (hdd/clj->vsa* [:. outcome :action]))
(defn sdm-outcome->output [outcome] (hdd/clj->vsa* [:. outcome :output]))
(defn sdm-outcome->target-state [outcome] (hdd/clj->vsa* [:. outcome :target-state]))

(comment
  (sdm/lookup sdm (hdd/clj->vsa* [:* :s0 0]) 3 1)
  (f/sum (read-1 sdm (hdd/clj->vsa* [:* :s0 0]) 3))
  60.0
  (for [branch [:action :output :target-state]]
    (hdd/clj->vsa*
     [:?? [:. (read-1 sdm (hdd/clj->vsa* [:* :s0 0]) 3) branch]]))
  (doseq [branch [:action :output :target-state]]
    (sdm/write
     sdm
     (hdd/clj->vsa* [:* :s0 0 branch])
     (hdd/clj->vsa*
      ({:action :right
        :output 0
        :target-state :s0}
       branch))
     1))
  (for [branch [:action :output :target-state]]
    (hdd/clj->vsa*
     [:?? (read-1 sdm (hdd/clj->vsa* [:* :s0 0 branch]) 1)])))

(defn ->sdm-fsm
  [quintuples]
  (let [sdm (sdm/->sdm {:address-count (long 1e6)
                        :address-density 0.000003
                        :word-length (long 1e4)})]
    (doseq [[state input-symbol target-state output-symbol
             action]
            quintuples]
      (sdm/write sdm
                 (hdd/clj->vsa* [:* state input-symbol])
                 (hdd/clj->vsa* (sdm-fsm-target
                                 target-state
                                 output-symbol
                                 action))
                 1))
    (reify
      FSM
      (transition [this current-state input-symbol]
        (read-1 sdm
                (hdd/clj->vsa* [:* current-state
                                input-symbol])
                3)))))



(comment
  ;; -----------------------------
  ;; Example data:
  ;;
  ;; This is a parity finder turing machine:
  ;;

  (def
    quintuples
    [[:s0 0    :s0    0 :right]
     [:s0 1    :s1    0 :right]
     [:s0 :b   :halt true :-]
     ;; -------------------------------
     [:s1 0 :s1 0 :right]
     [:s1 1 :s0 0 :right]
     [:s1 :b :halt false :-]])
  (def fsm (->sdm-fsm quintuples))
  (hdd/cleanup*
   (hdd/clj->vsa* [:. (transition fsm :s0 0) :output]))
  (hdd/cleanup*
   (hdd/clj->vsa* [:. (transition fsm :s0 1) :action])))


;; ----------------------

(defn iteratively-override
  [sdm address s top-k]
  ;;
  ;; I am under the impression that an implementation
  ;; in neuronal circuits is straightforward.
  ;;
  ;; Caveats
  ;;
  ;; I: 'current-content' register: My idea is that it
  ;; would be implemented with a stable neuronal
  ;; ensemble. Here, this is 'infinitely faithful' in
  ;; computer memory.
  ;;
  ;; II:
  ;; The notion of halting this subroutine surely must
  ;; be a best effort thing in a biological system.
  ;; (Perhaps with restarts etc.)
  ;;
  ;;
  (loop [current-content (read-1 sdm address top-k)]
    (if (and current-content
             (<= 0.95 (hd/similarity current-content s)))
      current-content
      (recur (do (sdm/write sdm address s 1)
                 (read-1 sdm address top-k))))))


(comment
  (iteratively-override sdm (hdd/clj->vsa* :foo) (hdd/clj->vsa* :bar) 1)
  (hd/similarity
   (hdd/clj->vsa* :bar)
   (read-1 sdm (hdd/clj->vsa* :foo) 1)))

(defn sdm-get-create
  ([sdm addr] (sdm-get-create sdm addr 1))
  ([sdm addr top-k]
   (let [content (read-1 sdm addr top-k)]
     ;;
     ;; This could be implemented by pre-allocated
     ;; random seeds.
     ;; (for instance neuronal ensembles created during
     ;; brain development) (G. Buzsáki)
     ;;
     ;; I.e. for everything that you ask for, you get
     ;; an answer (which can be a fresh random seed).
     ;;
     ;; Here, we allocate a random symbol the moment we
     ;; look instead. But I still feel entitled to call
     ;; it biologically principled.
     ;;
     (when-not content (sdm/write sdm addr (hd/->seed) 1))
     {:existed? (boolean content)
      :target (read-1 sdm addr top-k)})))

;; --------------------------

(comment
  (sdm/write sdm (hdd/clj->vsa* [:* head :right]) (hd/->seed) 1)
  (get-edge-create sdm head :right)
  (get-edge-create sdm head :left))


(comment
  (hd/similarity (hdd/clj->vsa* :a)
                 (flip (entangle [:a :b])
                       (hdd/clj->vsa* [:-- :b 0.5])))

  0.4)

(defn ->tape
  ([]
   (let [sdm (sdm/->sdm {:address-count (long 1e6)
                         :address-density 0.000003
                         :word-length (long 1e4)})
         head (atom (hd/->seed))
         directions-entanglement (entangle [:right :left])]
     (reify
       Tape
       ;; 2. tape element
       ;; address -> content
       ;;
       (write [this s]
         (iteratively-override sdm
                               @head
                               (hdd/clj->vsa* s)
                               1))
       (read-tape [this] (read-tape this 1))
       (read-tape [this top-k] (read-1 sdm @head top-k))
       (move [this direction] (move this direction 1))
       (move [this direction top-k]
         ;;
         ;; 1. edge:
         ;; current address * :right -> next-address
         ;;
         ;; To move, ask the system for the content
         ;; at addr address * :right, which is a
         ;; fresh random seed, if it doesn't exist
         ;;
         ;;
         ;; move:
         ;; 1. get/create current-addr * right ->
         ;; target (next-address)
         ;; 2. update head to target address
         ;; 3. handle backwards edge
         ;; The result is the tape head points to the
         ;; (possibly fresh) tile to the right
         ;;
         (let [current-head @head
               direction (hdd/clj->vsa* direction)
               {:keys [target]} (sdm-get-create
                                 sdm
                                 (hdd/clj->vsa*
                                  [:* current-head
                                   direction])
                                 ;; allow target to
                                 ;; be in
                                 ;; superposition
                                 top-k)
               ;; I also need to update the other
               ;; direction
               reverse-dir
               ;;
               ;; symbolically:
               ;; ({:left :right :right :left}
               ;; direction)
               ;;
               ;; hdc:
               ;;
               ;; Of course, the interesting thing
               ;; is that direction is allowed to
               ;; be a superposition of directions.
               (flip directions-entanglement direction)
               ;;
               ;; create the backward edge
               ;;
               ;;
               _ (sdm/write sdm
                            (hdd/clj->vsa* [:* target
                                            reverse-dir])
                            current-head
                            1)]
           (reset! head target))))))
  ([initial-tape]
   (let [res (->tape)]
     (doseq [symb initial-tape]
       (do (write res symb) (move res :right)))
     (doseq [_ initial-tape] (move res :left))
     res)))

;; ---------------------------------
;;

(comment
  (def tape (->tape [:a :b :c]))

  (do
    (def tape (->tape))
    (write tape :a)
    (move tape :right)
    (read-tape tape))
  nil

  (do
    (def tape (->tape))
    (write tape :a)
    (move tape :right)
    (move tape :left)
    (hdd/cleanup* (read-tape tape)))
  '(:a)

  (do
    (def tape (->tape [:a :b :c]))
    (write tape :a)
    (move tape :right)
    (move tape :left)
    (hdd/cleanup* (read-tape tape)))
  '(:a)

  (do
    (def tape (->tape [:a :b :c]))
    (write tape :a)
    (move tape :right)
    (move tape :right)
    (hdd/cleanup* (read-tape tape)))
  '(:c)

  (do
    (def tape (->tape [:a :b :c]))
    (write tape :a)
    (move tape :right)
    (move tape :right)
    (move tape :right)
    (read-tape tape))
  nil

  (do
    (def tape (->tape [:a :b :c]))
    (write tape :a)
    (move tape :right)
    (move tape :right)
    (move tape :right)
    (move tape :left)
    (move tape :left)
    (hdd/cleanup* (read-tape tape)))
  '(:b)


  (do
    (def tape (->tape [:a :b :c]))
    (write tape :a)
    (move tape :right)
    (move tape :right)
    (move tape :right)
    (move tape :left)
    (move tape :left)
    (move tape
          (hdd/clj->vsa* [:+ [:-- :right 0.5]
                          [:-- :left 0.5]])
          2)
    (hdd/cleanup* (read-tape tape 2)))
  '(:a :c)
  s)


(defn turing-machine
  [{:as turing-machine
    :keys [fsm-quintuples initial-tape initial-state
           max-steps]}]
  (reductions
    (fn [{:as machine
          :keys [tape fsm current-state input-symbol]} n]
      (let [outcome
              (transition fsm current-state input-symbol)
            action (sdm-outcome->action outcome)
            output (sdm-outcome->output outcome)
            target-state (sdm-outcome->target-state
                           outcome)]
        ;;
        ;; you can ask what is the :halt vote
        ;;
        (if (< 0.9
               (hd/similarity action (hdd/clj->vsa* :halt)))
          ;; Interesting problem comes up: When fsm is
          ;; in superposition, and the outcome of the
          ;; fsm contains 'halt', then what to halt?
          (ensure-reduced {:current-state target-state
                           :halted? true
                           :n n
                           :output output
                           :tape tape})
          (do (write tape output)
              (move tape action)
              (assoc machine
                :action action
                :input-symbol (read-tape tape)
                :current-state target-state)))))
    (let [tape (->tape initial-tape)]
      (assoc turing-machine
        :fsm (->sdm-fsm fsm-quintuples)
        :input-symbol (read-tape tape)
        :current-state (hdd/clj->vsa* initial-state)
        :tape tape))
    (if max-steps (range max-steps) (range))))


;; ----
;; Parity finder
;;

;; (from Minsky Finite and Infite Machines):
;;


(def outcomes
  (into []
        (turing-machine
         {:fsm-quintuples [[:s0 0 :s0 0      :right]
                           [:s0 1 :s1 0      :right]
                           [:s0 :b :halt true :halt]
                           ;; -------------------------------
                           [:s1 0 :s1 0 :right]
                           [:s1 1 :s0 0 :right]
                           [:s1 :b :halt false :halt]]
          :initial-state :s0
          :initial-tape [1 1 :b]})))

(hdd/cleanup* (:output (peek outcomes)))
'(true)

(def outcomes
  (into []
        (turing-machine
         {:fsm-quintuples [[:s0 0 :s0 0      :right]
                           [:s0 1 :s1 0      :right]
                           [:s0 :b :halt true :halt]
                           ;; -------------------------------
                           [:s1 0 :s1 0 :right]
                           [:s1 1 :s0 0 :right]
                           [:s1 :b :halt false :halt]]
          :initial-state :s0
          :initial-tape [1 1 0 1 :b]})))

(hdd/cleanup* (:output (peek outcomes)))
'(false)


(do
  (def outcomes
    (into []
          (turing-machine
           {:fsm-quintuples [[:s0 0 :s0 0      :right]
                             [:s0 1 :s1 0      :right]
                             [:s0 :b :halt true :halt]
                             ;; -------------------------------
                             [:s1 0 :s1 0 :right]
                             [:s1 1 :s0 0 :right]
                             [:s1 :b :halt false :halt]]
            :initial-state :s0
            :initial-tape [:b]})))
  (hdd/cleanup* (:output (peek outcomes))))
'(true)

(do (def outcomes
      (into []
            (turing-machine
             {:fsm-quintuples
              [[:s0 0 :s0 0 :right] [:s0 1 :s1 0 :right]
               [:s0 :b :halt true :halt]
               ;; -------------------------------
               [:s1 0 :s1 0 :right] [:s1 1 :s0 0 :right]
               [:s1 :b :halt false :halt]]
              :initial-state :s0
              :initial-tape [1 :b]})))
    (hdd/cleanup* (:output (peek outcomes))))
'(false)

(do (def outcomes
      (into []
            (turing-machine
             {:fsm-quintuples
              [[:s0 0 :s0 0 :right] [:s0 1 :s1 0 :right]
               [:s0 :b :halt true :halt]
               ;; -------------------------------
               [:s1 0 :s1 0 :right] [:s1 1 :s0 0 :right]
               [:s1 :b :halt false :halt]]
              :initial-state :s0
              :initial-tape [0 0 0 0 0 1 0 0 0 :b]})))
    (hdd/cleanup* (:output (peek outcomes))))
'(false)

(time
 (do (def outcomes
       (into []
             (turing-machine
              {:fsm-quintuples
               [[:s0 0 :s0 0 :right] [:s0 1 :s1 0 :right]
                [:s0 :b :halt true :halt]
                ;; -------------------------------
                [:s1 0 :s1 0 :right] [:s1 1 :s0 0 :right]
                [:s1 :b :halt false :halt]]
               :initial-state :s0
               :initial-tape [0 0 0 0 0 1 0 0 0 1 :b]})))
     (hdd/cleanup* (:output (peek outcomes)))))
'(true)
;; kinda slow impl
;; "Elapsed time: 743.824952 msecs"

(do (def outcomes
      (into []
            (turing-machine
             {:fsm-quintuples
              [[:s0 0 :s0 0 :right] [:s0 1 :s1 0 :right]
               [:s0 :b :halt true :halt]
               ;; -------------------------------
               [:s1 0 :s1 0 :right] [:s1 1 :s0 0 :right]
               [:s1 :b :halt false :halt]]
              :initial-state :s0
              :initial-tape [0 :b]})))
    (hdd/cleanup* (:output (peek outcomes))))
'(true)

(do (def outcomes
      (into []
            (turing-machine
             {:fsm-quintuples
              [[:s0 0 :s0 0 :right] [:s0 1 :s1 0 :right]
               [:s0 :b :halt true :halt]
               ;; -------------------------------
               [:s1 0 :s1 0 :right] [:s1 1 :s0 0 :right]
               [:s1 :b :halt false :halt]]
              :initial-state :s0
              :initial-tape [0 1 :b]})))
    (hdd/cleanup* (:output (peek outcomes))))
'(false)
(do (def outcomes
      (into []
            (turing-machine
             {:fsm-quintuples
              [[:s0 0 :s0 0 :right] [:s0 1 :s1 0 :right]
               [:s0 :b :halt true :halt]
               ;; -------------------------------
               [:s1 0 :s1 0 :right] [:s1 1 :s0 0 :right]
               [:s1 :b :halt false :halt]]
              :initial-state :s0
              :initial-tape [0 1 1 :b]})))
    (hdd/cleanup* (:output (peek outcomes))))
'(true)


;; --------------------------------------

;; 'halt' ideas:
;;
;; Idea 1:
;; - try 'remove' the 'halting' fsm from the
;;   superimposed fms, and keep around
;;   everything that halts as statistical
;;   outcome.
;; - but how to know what part of the fsm
;; resulted in the halting?
;; - one could re-run with less top-k and a
;;   little random seeds, thereby finding the
;;   cores of the fsm, then only continue
;;   with the cores that don't halt.
;; - this suggests a thought pump kind of
;;   search algorithm in the first place, run
;;   turing machine until 'halt' is part of
;;   the output, then narrow down from where
;;   the halt came?
;;
;; Idea 2:
;; - conventionally, make the halt output
;; simply keep the head of the tape
;; - :halt outcome also writes the output to
;; the tape
;; - ? :halt action as unit vector so it
;; stays automatically? (not sure if I can do
;; that with my block sparse codes).
;; - then run until the movement of the
;;   superimposed turing machines has settled
;;   to some specified degree.
;; - reading should then return the
;; superpositions of outcomes
;;
;; - the obvious cognitive connotation of
;; this is that you can either wait for a
;; majority to halt, then you get the
;; majority statistical vote of outcomes
;; ('system 1' kinda thing).
;;
;; - running longer gives you the long tail
;;   of 'thought trains'. *Selecting* in some
;;   way exactly *not* the large
;;   statistically overlapping outcomes, but
;;   *collapse* to some unlikely outcome
;;   (deep thought / system 2 ).
;; - Perhaps a mechanic for software
;; synthesis
;; - if you now *select* and make
;; this tail end the new statistically
;; likely. (Memetics)
;;
;; You see 2 memetic drivers, if we reward
;; all circuits that contribute to the
;; outcomes we *select*, then 1) finish very
;; fast, then you are part of the outcome in
;; mode 1 every time 2) be useful and deep,
;; dare I say perhaps creative and be
;; selected in mode 2
;;
;;
