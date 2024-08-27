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
           [:s1 0 :s1 0 :right]
           [:s1 1 :s0 0 :right]
           [:s1 :b :halt false :-]]]
    (hdd/fsa (for [[state input-symbol & outputs]
                     (hdd/clj->vsa* quintuples)]
               [state input-symbol
                (hdd/clj->vsa* (let [[state output action]
                                       outputs]
                                 {:action action
                                  :output output
                                  :state state}))]))))


;; ----

(def codebooks
  [(sdm/->sdm
    {:address-count (long 1e6)
     :address-density 0.000003
     :word-length (long 1e4)})
   (sdm/->sdm
    {:address-count (long 1e6)
     :address-density 0.000003
     :word-length (long 1e4)})
   (sdm/->sdm
    {:address-count (long 1e6)
     :address-density 0.000003
     :word-length (long 1e4)})])

(doall
 (map
  (fn [factors sdm]
    (doseq [f factors]
      (sdm/write sdm f f 1)))
  (map-indexed
   (fn [i factors] (map (fn [x] (hd/permute-n x i)) factors))
   (hdd/clj->vsa*
    [[:s0 :s1]
     [0 1 :halt false true]
     [:right :left :-]]))
  codebooks))

;; -------------

(def x (hdd/clj->vsa* [:*> :s0 0 :right]))

(defn bounce-resonator
  [codebooks x]
  (reductions
    (fn [{:keys [best-guesses confidence excitability]} n]
      (let [new-confidence
              (hd/similarity (hd/bind* best-guesses) x)
            excitability (max 5
                              (min 1
                                   (if (<= confidence
                                           new-confidence)
                                     (inc excitability)
                                     (dec excitability))))]
        (if (<= 0.99 confidence)
          (ensure-reduced {:best-guesses best-guesses
                           :confidence confidence
                           :n n})
          {:best-guesses (into []
                               (map (fn [sdm guess]
                                      (pyutils/torch->jvm
                                        (:result
                                          (sdm/lookup
                                            sdm
                                            guess
                                            excitability
                                            1))))
                                 codebooks
                                 best-guesses))
           :confidence confidence
           :n n})))
    {:best-guesses (for [sdm codebooks]
                     (pyutils/torch->jvm
                       (:result
                         (sdm/lookup sdm (hd/->ones) 4 1))))
     :confidence 0
     :excitability 4}
    (range 10)))

(bounce-resonator codebooks x)


(def best-guesses
  (into []
        (for [sdm codebooks]
          (pyutils/torch->jvm
           (:result (sdm/lookup sdm (hd/->ones) 4 1))))))

(map hdd/cleanup* (map-indexed (fn [i x] (hd/permute-n x (- i))) best-guesses))


'((:s0 :s1) (0 :halt true false) (:right :- :left))
'((:s0 :s1) (:halt true false) (:right :- :left))
'((:s0 :s1) (0 :halt true false) (:right :- :left))

(hdd/cleanup* (hd/unbind x (hd/bind* (rest best-guesses))))

(hd/similarity
 (hdd/clj->vsa* [:+ :s0 :s1])
 (hd/unbind x (hdd/clj->vsa*
               [:*
                [:> [:+ 0 1 true false]]
                [:>> [:+ :- :right :right]]])))

(hd/similarity
  (hd/bind* (into []
                  (for [sdm codebooks]
                    (pyutils/torch->jvm
                      (:result (sdm/lookup sdm
                                           (hd/drop-randomly
                                             (hd/->ones)
                                             0.5)
                                           1
                                           1))))))
  x)




(hd/similarity x (hd/bind* best-guesses))

(hd/similarity x
               (hdd/clj->vsa*
                [:*>
                 [:+ :s0 :s1]
                 [:+ 0 1 true false]
                 [:+ :- :right :right]]))

(hdd/cleanup* (first best-guesses))
'(:s0 :s1)
(hdd/cleanup* (hd/permute-inverse (second best-guesses)))
(hdd/cleanup* (hd/permute-inverse (hd/permute-inverse (second (rest best-guesses)))))





























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
