(ns fsm-and-tape
  (:refer-clojure :exclude [read])
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
;; read
;; write, grows left and right
;;

(defprotocol Tape
  (write [this s])
  (read [this])
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
  ;; 1. read current input symbol
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
        [[:s0 0 :s0 0 :right] [:s0 1 :s1 0 :right]
         [:s0 :b :halt true :-]
         ;; -------------------------------
         [:s1 0 :s1 0 :right] [:s1 1 :s0 0 :right]
         [:s1 :b :halt false :-]]]
    (hdd/fsa (for [[state input-symbol & outputs]
                   (hdd/clj->vsa* quintuples)]
               [state input-symbol
                (hdd/clj->vsa* (let [[state output action]
                                     outputs]
                                 {:action action
                                  :output output
                                  :state state}))]))))

(def action-decider
  (let [item-memory (hdd/->TinyItemMemory
                     {:left (hdd/clj->vsa* :left)
                      :right (hdd/clj->vsa* :right)})]
    (reify
      Decider
      (decide [this input]
        (prot/m-cleanup item-memory input)))))

(def state-decider
  (let [item-memory (hdd/->TinyItemMemory
                     (into {}
                           (for [k [:s0 :s1]]
                             [k (hdd/clj->vsa* k)])))]
    (reify
      Decider
      (decide [this input]
        (prot/m-cleanup item-memory input)))))

(defn ->tape [initial-tape]

  )

(defn execute
  [{:as turing-machine
    :keys [fsm initial-tape initial-state]}]
  (reductions
   (fn [{:keys [tape fsm state]}]
     (let [current-symbol (read tape)
           outcome (hdd/automaton-destination
                    fsm
                    current-symbol
                    state)
           action (decide action-decider
                          (hd/unbind outcome
                                     (hdd/clj->vsa*
                                      :action)))]
       (ensure-reduced [current-symbol outcome action])))
   turing-machine
   (assoc turing-machine
          :current-state initial-state
          :tape (->tape initial-tape))))









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
