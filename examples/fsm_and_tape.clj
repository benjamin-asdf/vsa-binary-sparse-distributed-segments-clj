(ns fsm-and-tape
  (:refer-clojure :exclude [read-tape])
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

;; Turing machine:
;; Finite State Machine and a Tape
;;

;; FSM, is a list of transition tuples:
;;
;; [old-state, input-symbol, new-state, output-symbol, action ]


;; tape:
;;
;; read-tape
;; write, grows left and right
;;

(defprotocol Tape
  (write [this s])
  (read-tape [this])
  (move [this direction]))


;; ----

(defprotocol FSM
  (transition [this current-state input-symbol]))

;; ----

(defprotocol Decider
  (decide [this input]))


;; ----



(defn turing-machine
  [{:keys [fsm initial-tape initial-state]}]

  ;;
  ;; 1. read-tape current input symbol
  ;; 2. use fsm
  ;;    obtain output symbol
  ;;    obtain action
  ;;    (obtain next state)
  ;;
  ;;
  ;; if action is halt, then stop
  ;;
  ;; 3. write output symbol (overriding current tape)
  ;; 4. move tape (this creates a new tile, if at the end of the tape)
  ;;
  ;;
  ;;
  ;;
  )

;; ----

(def fsa
  (let [quintuples
          [[:s0 0 :s0    0 :right]
           [:s0 1 :s1    0 :right]
           [:s0 :b :halt true :-]
           ;; -------------------------------
           [:s1 0 :s1 0 :right]
           [:s1 1 :s0 0 :right]
           [:s1 :b :halt false :-]]]
    (hdd/fsa (for [[state input-symbol & outputs]
                     (hdd/clj->vsa* quintuples)]
               [state input-symbol
                (hdd/clj->vsa*
                  (let [[next-state output action] outputs]
                    {:action action
                     :output output
                     :state next-state}))]))))

(comment
  (decide (->decider [:left :right :halt]) (hdd/clj->vsa* :left))
)
(defn ->decider
  [symbolic-codebook]
  (let [item-memory
        (hdd/->TinyItemMemory
         (atom
          (into {}
                (for [k symbolic-codebook]
                  [k (hdd/clj->vsa* k)]))))]
    (reify
      Decider
        (decide [this input]
          (prot/m-cleanup item-memory input)))))

(def action-decider (->decider [:left :right :halt :-]))
(def state-decider (->decider [:s1 :s0]))
(def output-symbol-decider (->decider [0 1 :b false true]))



;; ----------------------
















(def sdm (sdm/->sdm {:address-count (long 1e6)
                     :address-density 0.000003
                     :word-length (long 1e4)}))

(def head (hd/->seed))

(sdm/write sdm head (hdd/clj->vsa* :a) 1)

(sdm/converged-lookup-impl
 sdm
 head
 {:decoder-threshold 1
  :max-steps 5
  :stop? (fn [acc next-outcome]
           (when (< (:confidence next-outcome) 0.2)
             {:stop-reason :low-confidence
              :success? false}))
  :top-k 1})

(defn read-1
  [sdm addr top-k]
  (let [lookup-outcome (sdm/lookup sdm addr top-k 1)]
    (when (< 0.2 (:confidence lookup-outcome))
      (some-> lookup-outcome
              :result
              sdm/torch->jvm
              (dtt/->tensor :datatype :int8)))))

(defn iteratively-override
  [sdm address s top-k]
  (loop [current-content (read-1 sdm address top-k)]
    (if (and current-content
             (<= 0.95 (hd/similarity current-content s)))
      current-content
      (recur (do (sdm/write sdm address s 1)
                 (read-1 sdm address top-k))))))

(iteratively-override sdm (hdd/clj->vsa* :foo) (hdd/clj->vsa* :bar) 1)
(hd/similarity
 (hdd/clj->vsa* :bar)
 (read-1 sdm (hdd/clj->vsa* :foo) 1))


(defn sdm-get-create
  [sdm addr]
  (let [content (read-1 sdm addr 1)]
    (when-not content (sdm/write sdm addr (hd/->seed) 1))
    {:existed? (boolean content)
     :target (read-1 sdm addr 1)}))

(defn get-edge-create
  [sdm source direction]
  (let [edge (hdd/clj->vsa* [:* source direction])
        edge-destination (read-1 sdm edge 1)]
    (when-not edge-destination
      (sdm/write sdm edge (hd/->seed) 1))
    {:existed? (boolean edge-destination)
     :target (read-1 sdm edge 1)}))

(comment
  (sdm/write sdm (hdd/clj->vsa* [:* head :right]) (hd/->seed) 1)
  (get-edge-create sdm head :right)
  (get-edge-create sdm head :left))

(defn ->tape
  ([]
   (let [sdm (sdm/->sdm {:address-count (long 1e6)
                         :address-density 0.000003
                         :word-length (long 1e4)})
         head (atom (hd/->seed))]
     (reify
       Tape
         (write [this s]
           (iteratively-override sdm
                                 @head
                                 (hdd/clj->vsa* s)
                                 1))
         (read-tape [this] (read-1 sdm @head 1))
         (move [this direction]
           ;;
           ;; 1. edge:
           ;; current address * :right -> next-address
           ;;
           ;; 2. tape element
           ;; address -> content
           ;;
           ;;
           ;; moving right means checking if current
           ;; address * :right is stored in the sdm and
           ;; creating a new seed address, if not
           ;;
           ;;
           ;; move:
           ;; 1. create edge, if it doesn't exist
           ;; already
           ;; - current-addr * right -> next-address
           ;; - create a new seed for next-address, if
           ;; it doesn't exist
           ;; 2. update head to current-addr * right
           ;;
           (let [current-head @head
                 {:keys [target]} (sdm-get-create
                                    sdm
                                    (hdd/clj->vsa*
                                      [:* current-head
                                       direction]))
                 ;; for supporting :left, I also need
                 ;; to update the other direction if
                 ;; moving right, you need to make sure
                 ;; there is an edge new-addr * left ->
                 ;; old-addr
                 ;;
                 ;; if moving left, same thing for
                 ;; new-addr * right -> old-addr
                 reverse-dir ({:left :right :right :left}
                              direction)
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
  (move tape :left)
  (hdd/cleanup (read-tape tape))

  (def tape (->tape))
  (write tape :a)
  (hdd/cleanup (read-tape tape))
  (move tape :right)
  (write tape :b)
  (hdd/cleanup (read-tape tape))
  (move tape :left))

(defn execute
  [{:as turing-machine
    :keys [fsm initial-tape initial-state]}]
  (reductions
    (fn [{:as machine :keys [tape fsm state input-symbol]}
         n]
      (let [_ (def fsm fsm)
            _ (def input-symbol input-symbol)
            _ (def state state)
            outcome (hdd/automaton-destination
                     fsm
                     ;; is ~ the same as having a
                     ;; cleaned up hdv at hand
                     ;;
                     (hdd/clj->vsa* state)
                     (hdd/clj->vsa* input-symbol))
            _ (def outcome
                (hdd/automaton-destination
                 fsm
                 ;; is ~ the same as having a cleaned
                 ;; up hdv at hand
                 ;;
                 (hdd/clj->vsa* input-symbol)
                 (hdd/clj->vsa* state)))
            ;; _ (throw (Exception. "foo"))
            action (decide action-decider
                           (hd/unbind outcome
                                      (hdd/clj->vsa*
                                       :action)))
            _ (when-not action
                (throw (Exception. "foo")))
            output (decide output-symbol-decider
                           (hd/unbind outcome
                                      (hdd/clj->vsa*
                                       :output)))
            next-state (decide state-decider
                               (hd/unbind outcome
                                          (hdd/clj->vsa*
                                           :state)))]
        (if (= action :halt)
          (ensure-reduced
            {:halted? true :n n :output output :tape tape})
          (do (write tape output)
              (move tape action)
              (assoc
               machine
               :action action
               :input-symbol (read-tape tape)
               :state next-state)))))
    (let [tape (->tape initial-tape)]
      (assoc turing-machine
        :input-symbol (read-tape tape)
        :state initial-state
        :tape tape))
    (range 5)))

(hdd/cleanup input-symbol)

state

(hd/similarity
 (hdd/clj->vsa* :action)
 (hd/unbind
  (hdd/automaton-destination fsm
                             (hdd/clj->vsa* state)
                             (hdd/clj->vsa* input-symbol))
  (hdd/clj->vsa* :action)))


(hdd/cleanup-verbose
 (hd/unbind
  (hdd/automaton-destination fsm
                             (hdd/clj->vsa* state)
                             (hdd/clj->vsa* input-symbol))
  (hdd/clj->vsa* :state)))
(hdd/cleanup-verbose
 (hd/unbind
  (hdd/automaton-destination fsm
                             (hdd/clj->vsa* state)
                             (hdd/clj->vsa* input-symbol))
  (hdd/clj->vsa* :output)))



;; ----------------------

(execute
 {:fsm fsa :initial-state :s0 :initial-tape [0 1 1 1 0 :b]})

(def outcome *1)

(map hdd/cleanup (keep :input-symbol (take 5 outcome)))
(0 1 0 nil)
(0 1 0 nil)
(map hdd/cleanup (keep :action (take 5 outcome)))



(hdd/cleanup (hd/unbind outcome (hdd/clj->vsa* :output)))







(hdd/cleanup
 (hdd/clj->vsa* [:. (hdd/automaton-destination fsa (hdd/clj->vsa :s0) (hdd/clj->vsa 0)) :action]))


;; ----
;; Parity finder
;;

;; (from Minsky Finite and Infite Machines):
;;

(turing-machine
 {:fsm
  [[:s0 0 :s0 0 :right]
   [:s0 1 :s1 0 :right]
   [:s0 :b :halt true nil]
   ;; -------------------------------
   [:s1 0 :s1 0 :right]
   [:s1 1 :s0 0 :right]
   [:s1 :b :halt false nil]]
  :initial-tape
  [0 1 1 0 1 :b]
  :initial-state :s0})

;; ------------























(comment
  (for [state [:s0 :s1]
        input [0 1 :b]
        q [:state :action :output]]
    [state input :q q
     (hdd/cleanup (hd/unbind (hdd/automaton-destination
                              fsa
                              (hdd/clj->vsa* state)
                              (hdd/clj->vsa* input))
                             (hdd/clj->vsa* q)))])
  '([:s0 0 :q :state :s0]
    [:s0 0 :q :action :right]
    [:s0 0 :q :output 0]
    [:s0 1 :q :state :s1]
    [:s0 1 :q :action :right]
    [:s0 1 :q :output 0]
    [:s0 :b :q :state :halt]
    [:s0 :b :q :action :-]
    [:s0 :b :q :output true]
    [:s1 0 :q :state :s1]
    [:s1 0 :q :action :right]
    [:s1 0 :q :output 0]
    [:s1 1 :q :state :s0]
    [:s1 1 :q :action :right]
    [:s1 1 :q :output 0]
    [:s1 :b :q :state :halt]
    [:s1 :b :q :action :-]
    [:s1 :b :q :output false]))
