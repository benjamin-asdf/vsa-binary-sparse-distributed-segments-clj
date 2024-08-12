(ns what-is-the-bread-for-lava
  (:require
   [tech.v3.datatype.functional :as f]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [tech.v3.datatype.bitmap :as bitmap]
   [fastmath.random :as fm.rand]
   [fastmath.core :as fm]
   [bennischwerdtner.sdm.sdm :as sdm]
   [bennischwerdtner.hd.binary-sparse-segmented :as
    hd]
   [bennischwerdtner.pyutils :as pyutils]
   [tech.v3.datatype.unary-pred :as unary-pred]
   [tech.v3.datatype.argops :as dtype-argops]
   [bennischwerdtner.hd.data :as hdd]))

;; -----------------
;;
;; Pretend for a moment you have a cognitive system that already established certain 'samenesses'
;;

(def liquid (hdd/clj->vsa :liquid))
;; forgot about the honey
;; (def honey (hd/thin (hd/superposition liquid (hdd/clj->vsa :honey))))
(def butter (hd/thin (hd/superposition liquid (hdd/clj->vsa :butter))))
(def lava (hd/thin (hd/superposition liquid (hdd/clj->vsa :lava))))


;; -------------------------------------------------------
;; Model stuff with finite state automatons
;;
;; Say that the outcome of spread is a surface filled with *something*
;;
;;

(def bread-domain
  (hdd/finite-state-automaton-1
    (hdd/clj->vsa* [[:bread :spread {:surface butter}]
                    [:bread :spread {:surface :vegan-spread}]
                    [:bread :crumble {:crumps :bread}]
                    [:bread :forget {:bread :molded}]])))
(def lava-domain
  (hdd/finite-state-automaton-1
    (hdd/clj->vsa* [[:rocks :spread {:surface lava}]
                    [:lava :freeze :rocks]
                    [:vulcano :erupt {:spew lava}]
                    [:rocks :forget {:rocks :ancient}]])))


;; what is the bread for lava?

;; first ask what is the action I would do with lava and bread,
;; the similarity of lava and liquid doing work now, the shared :spread comes out:

(hdd/cleanup*
 (hd/unbind
  bread-domain
  (hd/bind
   (hdd/clj->vsa* :bread)
   (hd/permute (hdd/clj->vsa* {:surface lava})))))
'(:spread)


;; the essential mechanism for this is at fun_with_trees.clj

(let
    [the-action-that-would-lead-to-lava-surface-given-a-bread
     (hd/unbind bread-domain
                (hd/bind (hdd/clj->vsa* :bread)
                         (hd/permute (hdd/clj->vsa*
                                      {:surface lava}))))]
    [:spread-lava-in-bread-domain
     (hdd/cleanup*
      (hdd/automaton-source
       bread-domain
       the-action-that-would-lead-to-lava-surface-given-a-bread
       (hdd/clj->vsa* {:surface lava})))
     :spread-lava-in-bread+lava-domain
     (hdd/cleanup*
      (hdd/automaton-source
       (hdd/union bread-domain lava-domain)
       ;; for this to work, I need need to cleanup with
       ;; an item memory
       ;; (literature mentions this as challange.
       ;; Resonator networks can do this efficiently)
       ;; --------------------------------------------
       ;; cleanup to prestine :spread
       (hdd/clj->vsa
        (hdd/cleanup
         the-action-that-would-lead-to-lava-surface-given-a-bread))
       (hdd/clj->vsa* {:surface lava})))])

[:spread-lava-in-bread-domain '(:bread)
 :spread-lava-in-bread+lava-domain '(:rocks :bread)]






;; -----------------------------------------------------------




;;
;; in some ways you expect something that is now 'superimposed with bread'?
;; Perhaps it is exactly this superposition that is the early making of a personal 'inside joke'
;; The concept that bread and lava now coexist?
;; I don't know.
;;

;; -----------
;; Admittedly, :spread is doing a lot of work here.
;;

;; -----------------------------------------------------------

(=
 (hdd/finite-state-automaton-1
  (hdd/clj->vsa* [[:bread :spread {:surface butter}]]))
 ;; -------------------------------
 ;; expands to
 (hd/bind
  (hd/bind (hdd/clj->vsa* :bread) (hdd/clj->vsa* :spread))
  (hd/permute (hd/bind (hdd/clj->vsa* :surface) butter))))
true

(=
 (hdd/automaton-source
  (hd/bind (hd/bind (hdd/clj->vsa* :bread)
                    (hdd/clj->vsa* :spread))
           (hd/permute (hd/bind (hdd/clj->vsa* :surface)
                                butter)))
  (hdd/clj->vsa* :spread)
  (hd/bind (hdd/clj->vsa* :surface) butter))
 (hd/unbind
  (hd/bind (hd/bind (hdd/clj->vsa* :bread)
                    (hdd/clj->vsa* :spread))
           (hd/permute (hd/bind (hdd/clj->vsa* :surface)
                                butter)))
  (hd/bind (hdd/clj->vsa* :spread)
           (hd/permute (hd/bind (hdd/clj->vsa* :surface)
                                butter))))
 ;; ---------------------------
 ;; The user can work this out for themselves, it becomes:
 (hdd/clj->vsa* :bread))
true


;; ...
;; if you put a superposition instead of bread:

(= (hdd/automaton-source
    ;; Flipping the args here for the subtle reason
    ;; that dense hdvs don't actually have a
    ;; commutative bind. (The
    ;; bit count of the second arg is preserved)
    ;; This affects the outcome of '='
    (hd/bind (hd/permute (hd/bind (hdd/clj->vsa* :surface)
                                  butter))
             (hd/bind (hdd/clj->vsa* :spread)
                      (hdd/clj->vsa* #{:bread :rocks})))
    (hdd/clj->vsa* :spread)
    (hd/bind (hdd/clj->vsa* :surface) butter))
   ;; --------------------------------------------------
   ;; expanded:
   (hd/unbind
    (hd/bind (hd/permute (hd/bind (hdd/clj->vsa* :surface)
                                  butter))
             (hd/bind (hdd/clj->vsa* :spread)
                      (hdd/clj->vsa* #{:bread :rocks})))
    (hd/bind (hdd/clj->vsa* :spread)
             (hd/permute (hd/bind (hdd/clj->vsa* :surface)
                                  butter))))
   ;; --------------------------------------------------
   ;; ... then a superposition comes out here:
   ;;
   (hdd/clj->vsa* #{:bread :rocks}))
true

;; ----------------------------------------
;;

;;
;; (⊕ bread-domain lava-domain )
;;
;; I use #{} interchangibly with ⊕
;;
;;
;; comes down essentially to
;;
;;  #{:bread :rocks} ⊙ :spread ⊙ ~ p(:surface ⊙ :liquid)
;;
;;
;; Then querying with as destination {:surface lava} and :spread as input token
;;
;;
;; a = #{:bread :rocks} ⊙ :spread ⊙ ~ p(:surface ⊙ :liquid)
;;
;; a ⊘ ( :spread ⊙ p( ~ {:suface :liquid} ) )
;;
;; -> #{:bread :rocks}
;;

;; ------------------------------------------------

;; more mechanism:

(hdd/cleanup-verbose
 (hd/unbind
  (hdd/automaton-destination
   ;; you see that such an automaton supports 2
   ;; paths towards ~ {:surface liquid}
   (hdd/finite-state-automaton-1
    (hdd/clj->vsa*
     [[:rocks :spread {:surface #{:butter :liquid}}]
      [:bread :spread {:surface #{:lava :liquid}}]]))
   (hdd/clj->vsa* :spread)
   (hdd/clj->vsa* #{:rocks :bread}))
  (hdd/clj->vsa* :surface)))

;; ({:k :liquid
;;   :similarity 0.55
;;   :v #tech.v3.tensor<int8> [10000]
;;   [0 0 0 ... 0 0 0]}
;;  {:k :lava
;;   :similarity 0.3
;;   :v #tech.v3.tensor<int8> [10000]
;;   [0 0 0 ... 0 0 0]}
;;  {:k :butter
;;   :similarity 0.15
;;   :v #tech.v3.tensor<int8> [10000]
;;   [0 0 0 ... 0 0 0]})

(hdd/cleanup-verbose
 (hdd/automaton-source
  ;; you see that such an automaton supports 2
  ;; paths towards ~ {:surface liquid}
  (hdd/finite-state-automaton-1
   (hdd/clj->vsa*
    [[:rocks :spread {:surface butter}]
     [:bread :spread {:surface lava}]]))
  (hdd/clj->vsa* :spread)
  (hdd/clj->vsa* {:surface liquid})))

;; ({:k :rocks
;;   :similarity 0.6
;;   :v #tech.v3.tensor<int8> [10000]
;;   [0 0 0 ... 0 0 0]}
;;  {:k :bread
;;   :similarity 0.45
;;   :v #tech.v3.tensor<int8> [10000]
;;   [0 0 0 ... 0 0 0]})

(hdd/cleanup-verbose
 (hdd/automaton-source
  ;; you see that such an automaton supports 2
  ;; paths towards ~ {:surface liquid}
  (hdd/finite-state-automaton-1
   (hdd/clj->vsa*
    [[:rocks :spread {:surface butter}]
     [:bread :spread {:surface lava}]]))
  (hdd/clj->vsa* :spread)
  (hdd/clj->vsa* {:surface lava})))

;; ({:k :bread
;;   :similarity 1.0
;;   :v #tech.v3.tensor<int8> [10000]
;;   [0 0 0 ... 0 0 0]}
;;  {:k :rocks
;;   :similarity 0.3
;;   :v #tech.v3.tensor<int8> [10000]
;;   [0 0 0 ... 0 0 0]})

(hdd/cleanup-verbose
 (hdd/automaton-source
  ;; you see that such an automaton supports 2
  ;; paths towards ~ {:surface liquid}
  (hd/superposition
   (hdd/finite-state-automaton-1
    (hdd/clj->vsa*
     [[:rocks :spread {:surface butter}]]))
   (hdd/finite-state-automaton-1
    (hdd/clj->vsa*
     [[:bread :spread {:surface lava}]])))
  (hdd/clj->vsa* :spread)
  (hdd/clj->vsa* {:surface lava})))

;; ({:k :bread
;;   :similarity 1.0
;;   :v #tech.v3.tensor<int8> [10000]
;;   [0 0 0 ... 0 0 0]}
;;  {:k :rocks
;;   :similarity 0.3
;;   :v #tech.v3.tensor<int8> [10000]
;;   [0 0 0 ... 0 0 0]})


;; -------------------------------------------------------
;; Model as finite state automaton
;; -------------------------------------------------------
;; - this is exploratory, premilinary
;;


;; -----

;; We need to be able to cleanup composits for this to work, use an sdm as item memory
;; (serves as an example usage of sdm)
;;

(defprotocol ItemMemory
  (remember [this addr])
  (recover [this addr-prime]))

(def item-memory
  (let [sdm (sdm/->sdm {:address-count (long 1e5)
                        :address-density 0.00003
                        :word-length (long 1e4)})]
    (reify
      ItemMemory
        (remember [this addr] (sdm/write sdm addr addr 1))
        (recover [this addr-prime]
          ;; didn't figure out the interface yet
          ;; so I specify it in user space here.
          (some->
            (sdm/converged-lookup-impl
              sdm
              addr-prime
              {:decoder-threshold 1
               :stop? (fn [acc next-outcome]
                        (cond
                          (< 0.9 (:confidence next-outcome))
                            {:stop-reason :high-confidence
                             :success? true}
                          ;;
                          ;; two times low
                          ;; confidence, means
                          ;; it's diverging I
                          ;; think
                          ;;
                          (when-let [last-confidence
                                       (:confidence
                                         (peek (:result-xs
                                                 acc)))]
                            (< (+ (:confidence next-outcome)
                                  last-confidence)
                               0.2))
                            {:stop-reason :diverging
                             :success? false}))
               ;; higher top-k so I get overlap with
               ;; compositis
               ;; of 'order 2' (hereby definend as sort
               ;; roughly the multiple of contributing
               ;; seed vectors)
               :top-k 2})
            :result-address
            sdm/torch->jvm
            (dtt/->tensor :datatype :int8))))))


;; -----------

(defn hdv-nodes
  [xs]
  (->> (tree-seq (fn [x] (and (not (hd/hv? x)) (seq? x)))
                 seq
                 xs)
       (filter hd/hv?)))

(defn remember-leaves!
  [xs]
  (doseq [item (hdv-nodes xs)]
    (remember item-memory item))
  xs)

;; -----------

;; Model the bread domain as a finite state automaton.

(def bread-domain
  (hdd/finite-state-automaton-1
    (remember-leaves!
      (hdd/clj->vsa*
       [[{:bread {:surface :empty}}
         {:spread butter}
         {:bread {:surface butter}}]
         [{:bread {:surface butter}} {:spread butter}
          {:bread {:surface butter :thick? true}}]
         [{:bread {:surface butter}} {:scrub-off butter}
          {:bread {:surface butter :thin? true}}]]))))



;; -----------------------------------------------------------
;; Update: Using clj->vsa* dsl

(let
  [the-action-that-would-lead-to-lava-surface-given-a-bread
     (hdd/clj->vsa*
      [:*.< bread-domain :bread :_ {:surface lava}])]
  [:spread-lava-in-bread-domain
   (hdd/cleanup*
    (hdd/clj->vsa*
     [:*.< bread-domain :_ the-action-that-would-lead-to-lava-surface-given-a-bread
      {:surface lava}]))
   :spread-lava-in-bread+lava-domain
   (hdd/cleanup*
    (hdd/clj->vsa*
     [:*.<
      [:+ bread-domain lava-domain]
      :_
      (hdd/clj->vsa (hdd/cleanup the-action-that-would-lead-to-lava-surface-given-a-bread))
      {:surface lava}]))])

'[:spread-lava-in-bread-domain (:bread)
  :spread-lava-in-bread+lava-domain (:rocks :bread)]

;; ... Arguably not better, lol
;;



;; -----------------------------------------
;; Appendix: Checking some properties of the sdm in usage
;;

(comment
  (let [item-memory
        (let [sdm (sdm/->sdm {:address-count (long 1e5)
                              :address-density 0.00003
                              :word-length (long 1e4)})]
          (reify
            ItemMemory
            (remember [this addr]
              (sdm/write sdm addr addr 1))
            (recover [this addr-prime]
              ;; didn't figure out the interface yet
              ;; so I specify it in user space here.
              (some->
               (sdm/converged-lookup-impl
                sdm
                addr-prime
                {:decoder-threshold 1
                 :stop?
                 (fn [acc next-outcome]
                   (cond
                     (< 0.9
                        (:confidence next-outcome))
                     {:stop-reason :high-confidence
                      :success? true}
                     ;;
                     ;; two times low
                     ;; confidence, means
                     ;; it's diverging I
                     ;; think
                     ;;
                     (when-let [last-confidence
                                (:confidence (peek (:result-xs acc)))]
                       (< (+ (:confidence next-outcome)
                             last-confidence)
                          0.2))
                     {:stop-reason :diverging
                      :success? false}))
                 :top-k 1})
               :result-address
               sdm/torch->jvm
               (dtt/->tensor :datatype :int8)))))]
    (remember item-memory (hdd/clj->vsa :a))
    (doall
     (for
         [n (range 1000)]
         (remember item-memory (hd/->seed))))
    (hdd/cleanup* (recover item-memory (hdd/clj->vsa :a))))
  '(:a)

  (let [item-memory
        (let [sdm (sdm/->sdm {:address-count (long 1e5)
                              :address-density 0.00003
                              :word-length (long 1e4)})]
          (reify
            ItemMemory
            (remember [this addr]
              (sdm/write sdm addr addr 1))
            (recover [this addr-prime]
              ;; didn't figure out the interface yet
              ;; so I specify it in user space here.
              (some->
               (sdm/converged-lookup-impl
                sdm
                addr-prime
                {:decoder-threshold 1
                 :stop?
                 (fn [acc next-outcome]
                   (cond
                     (< 0.9
                        (:confidence next-outcome))
                     {:stop-reason :high-confidence
                      :success? true}
                     ;;
                     ;; two times low
                     ;; confidence, means
                     ;; it's diverging I
                     ;; think
                     ;;
                     (when-let [last-confidence
                                (:confidence
                                 (peek (:result-xs
                                        acc)))]
                       (< (+ (:confidence
                              next-outcome)
                             last-confidence)
                          0.2))
                     {:stop-reason :diverging
                      :success? false}))
                 :top-k 1})
               :result-address
               sdm/torch->jvm
               (dtt/->tensor :datatype :int8)))))]
    (remember item-memory (hdd/clj->vsa :a))
    (map (fn [x]
           (some->> (recover item-memory x)
                    (hdd/cleanup*)))
         [(hdd/clj->vsa :a) (hd/weaken (hdd/clj->vsa :a) 0.5)
          (hd/weaken (hdd/clj->vsa :a) 0.75)
          (hd/weaken (hdd/clj->vsa :a) 1)]))
  '((:a) (:a) (:a) nil)

  ;; ------------------
  ;; T = 1000
  ;; (test data set count)
  ;;

  (def outcome
    (doall
     (for [n (range 5)]
       (let [item-memory
             (let [sdm (sdm/->sdm {:address-count (long 1e5)
                                   :address-density 0.00003
                                   :word-length (long 1e4)})]
               (reify
                 ItemMemory
                 (remember [this addr]
                   (sdm/write sdm addr addr 1))
                 (recover [this addr-prime]
                   ;; didn't figure out the interface yet
                   ;; so I specify it in user space here.
                   (some->
                    (sdm/converged-lookup-impl
                     sdm
                     addr-prime
                     {:decoder-threshold 1
                      :stop?
                      (fn [acc next-outcome]
                        (def acc acc)
                        (def next-outcome next-outcome)
                        (cond
                          (< 0.9
                             (:confidence next-outcome))
                          {:stop-reason :high-confidence
                           :success? true}
                          ;;
                          ;; two times low
                          ;; confidence, means
                          ;; it's diverging I
                          ;; think
                          ;;
                          (when-let [last-confidence
                                     (:confidence
                                      (peek (:result-xs
                                             acc)))]
                            (< (+ (:confidence
                                   next-outcome)
                                  last-confidence)
                               0.2))
                          {:stop-reason :diverging
                           :success? false}))
                      :top-k 1})
                    :result-address
                    sdm/torch->jvm
                    (dtt/->tensor :datatype :int8)))))]
         (remember item-memory (hdd/clj->vsa :a))
         (doall (for [n (range 1000)]
                  (remember item-memory (hd/->seed))))
         (doall
          (for [n [5 7 10 15]]
            (let [x (hd/thin (apply hd/superposition
                                    (concat [(hdd/clj->vsa :a)]
                                            (repeatedly n #(hd/->hv)))))]
              {:noise-factor n
               :recovered? (= '(:a)
                              (some->> (recover item-memory x)
                                       (hdd/cleanup*)))})))))))

  ;; superposition with 5 random vectors and thinning
  ;; recovered only 3/5 times

  '(({:noise-factor 5 :recovered? true}
     {:noise-factor 7 :recovered? false}
     {:noise-factor 10 :recovered? false}
     {:noise-factor 15 :recovered? false})
    ({:noise-factor 5 :recovered? true}
     {:noise-factor 7 :recovered? false}
     {:noise-factor 10 :recovered? true}
     {:noise-factor 15 :recovered? false})
    ({:noise-factor 5 :recovered? false}
     {:noise-factor 7 :recovered? false}
     {:noise-factor 10 :recovered? true}
     {:noise-factor 15 :recovered? false})
    ({:noise-factor 5 :recovered? false}
     {:noise-factor 7 :recovered? false}
     {:noise-factor 10 :recovered? true}
     {:noise-factor 15 :recovered? false})
    ({:noise-factor 5 :recovered? false}
     {:noise-factor 7 :recovered? true}
     {:noise-factor 10 :recovered? false}
     {:noise-factor 15 :recovered? false}))



  ;; -------------------------------------------
  ;; without thinning

  (def outcome
    (doall
     (for [n (range 5)]
       (let [item-memory
             (let [sdm (sdm/->sdm {:address-count (long 1e5)
                                   :address-density 0.00003
                                   :word-length (long 1e4)})]
               (reify
                 ItemMemory
                 (remember [this addr]
                   (sdm/write sdm addr addr 1))
                 (recover [this addr-prime]
                   ;; didn't figure out the interface yet
                   ;; so I specify it in user space here.
                   (some->
                    (sdm/converged-lookup-impl
                     sdm
                     addr-prime
                     {:decoder-threshold 1
                      :stop?
                      (fn [acc next-outcome]
                        (cond
                          (< 0.9
                             (:confidence next-outcome))
                          {:stop-reason :high-confidence
                           :success? true}
                          ;;
                          ;; two times low
                          ;; confidence, means
                          ;; it's diverging I
                          ;; think
                          ;;
                          (when-let [last-confidence
                                     (:confidence
                                      (peek (:result-xs
                                             acc)))]
                            (< (+ (:confidence
                                   next-outcome)
                                  last-confidence)
                               0.2))
                          {:stop-reason :diverging
                           :success? false}))
                      :top-k 1})
                    :result-address
                    sdm/torch->jvm
                    (dtt/->tensor :datatype :int8)))))]
         (remember item-memory (hdd/clj->vsa :a))
         (doall (for [n (range 1000)]
                  (remember item-memory (hd/->seed))))
         (doall
          (for [n [5 7 10 15]]
            (let [x (apply hd/superposition
                           (concat [(hdd/clj->vsa :a)]
                                   (repeatedly n #(hd/->hv))))]
              {:noise-factor n
               :recovered? (= '(:a)
                              (some->> (recover item-memory x)
                                       (hdd/cleanup*)))})))))))


  ;; if we you don't thin, you can query and recover in this config

  (({:noise-factor 5 :recovered? true}
    {:noise-factor 7 :recovered? true}
    {:noise-factor 10 :recovered? true}
    {:noise-factor 15 :recovered? true})
   ({:noise-factor 5 :recovered? true}
    {:noise-factor 7 :recovered? true}
    {:noise-factor 10 :recovered? true}
    {:noise-factor 15 :recovered? true})
   ({:noise-factor 5 :recovered? true}
    {:noise-factor 7 :recovered? true}
    {:noise-factor 10 :recovered? true}
    {:noise-factor 15 :recovered? true})
   ({:noise-factor 5 :recovered? true}
    {:noise-factor 7 :recovered? true}
    {:noise-factor 10 :recovered? true}
    {:noise-factor 15 :recovered? true})
   ({:noise-factor 5 :recovered? true}
    {:noise-factor 7 :recovered? true}
    {:noise-factor 10 :recovered? true}
    {:noise-factor 15 :recovered? true})))
