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

;; Get a mix:
(hdd/cleanup*
 (hdd/automaton-source
  (hd/superposition bread-domain lava-domain)
  (hdd/clj->vsa* :spread)
  (hdd/clj->vsa* {:surface lava})))
'(:rocks :bread)

;; When you know that you aren't looking for bread:
(hdd/cleanup*
 (hdd/automaton-source
  lava-domain
  (hdd/clj->vsa* :spread)
  (hdd/clj->vsa* {:surface lava})))
'(:rocks)
;; (kinda shows that the :spread domain in truth is doing the lifting)


;; Or remove it from the outcome
(hdd/cleanup*
 (hdd/difference
  (hdd/automaton-source
   (hd/superposition bread-domain
                     lava-domain)
   (hdd/clj->vsa* :spread)
   (hdd/clj->vsa* {:surface lava}))
  (hdd/clj->vsa* :bread)))
'(:rocks)


;;
;; in some ways you expect something that is now 'superimposed with bread'?
;; Perhaps it is exactly this superposition that is the early making of a personal 'inside joke'
;; The concept that bread and lava now coexist?
;; I don't know.
;;

;; -----------
;; Admittedly, :spread is doing a lot of work here.
;;
;; -----------

;; ----------------------------------
;;

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

;; ({:k :bread
;;   :similarity 0.5
;;   :v #tech.v3.tensor<int8> [10000]
;;   [0 0 0 ... 0 0 0]}
;;  {:k :rocks
;;   :similarity 0.4
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
;;   :similarity 0.25
;;   :v #tech.v3.tensor<int8> [10000]
;;   [0 0 0 ... 0 0 0]})

;; ({:k :bread
;;   :similarity 1.0
;;   :v #tech.v3.tensor<int8> [10000]
;;   [0 0 0 ... 0 0 0]}
;;  {:k :rocks
;;   :similarity 0.25
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
;;   :similarity 0.25
;;   :v #tech.v3.tensor<int8> [10000]
;;   [0 0 0 ... 0 0 0]})


;; -------------------------------------------------------------





;; to show that you can juggle these things a little,
;; let's query the union:

(hdd/cleanup*
 (hd/unbind
  ;; ~ {:ground (⊕ :rocks :bread)}
  (hd/unbind (hdd/union rocks-domain bread-domain)
             ;; intersection
             ;; ~ {:surface liquid}
             (hdd/intersection-1 [bread-domain
                                rocks-domain]))
  (hdd/clj->vsa* :ground)))
'(:bread :rocks)


;; the difference is also meaningful
(hdd/cleanup*
 (hd/unbind
  (hd/unbind (hdd/difference rocks-domain bread-domain)
             ;; intersection
             ;; ~ {:surface liquid}
             (hdd/intersectiuon [bread-domain
                                rocks-domain]))
  (hdd/clj->vsa* :ground)))
'(:rocks)

;; other way around:
(hdd/cleanup*
 (hd/unbind
  (hd/unbind (hdd/difference bread-domain rocks-domain)
             ;; intersection
             ;; ~ {:surface liquid}
             (hdd/intersection-1 [bread-domain
                                rocks-domain]))
  (hdd/clj->vsa* :ground)))
'(:bread)


;; ---------------------------------
;; Showing that this really has to do with 'liquid':
;;

(def tofifee-domain
  (hdd/clj->vsa* {:ground :caramel :surface :chocolate}))

(hdd/cleanup*
 (hd/unbind
  ;; ~ nothing
  (hd/unbind rocks-domain
             ;; intersection
             ;; ~ nothing !
             (hdd/intersection-1 [bread-domain tofifee-domain]))
  (hdd/clj->vsa* :ground)))

(f/sum (hdd/intersection-1 [bread-domain tofifee-domain]))

(= (hdd/clj->vsa* :foo) (hd/unbind (hdd/clj->vsa* :foo) (hd/->empty)))



;; ------------------
;; Partial mathematical mechanism:
;;

(comment
  (f/sum (hdd/intersection-1 [bread-domain tofifee-domain]))
  0.0
  ;; 5 bits is what we needed
  (f/sum (hdd/intersection-1 [bread-domain rocks-domain]))
  5.0
  ;; (this exact outcome is non deterministic, depends
  ;; on the seed vectors)
  ;; since segment-count = 20, and both
  ;; [(⊕ :lava liquid)] ~
  ;; [(⊕ :butter liquid)] ~  0.5 similarity to liquid
  ;; (both have ~10 liquid bits)
  ;; -------------------------------------------------
  ;; [ . ] means normalizing, here that is [[hd/thin]]
  ;; -------------------------------------------------
  ;; the count of overlap between butter and lava is
  ;; roughly 5:
  ;; -------------------------------------------------
  ;; O((∩ butter lava)) ~= 5
  ;;
  ;;
  ;; ... and those 5 bits happen to be mapped to the
  ;; :surface domain so to say.
  ;;
  (hd/similarity (hdd/clj->vsa* {:surface liquid})
                 (hdd/intersection-1 [bread-domain
                                    rocks-domain]))
  ;; similarity > 0.1 usually means 'similar' here
  0.25
  ;; It happens to be the case that all 5 bits that
  ;; make up the similarity
  ;; are also part of {:surface liquid}
  (f/sum (hdd/intersection-1
          [(hdd/clj->vsa* {:surface liquid})
           (hdd/intersection-1 [bread-domain
                              rocks-domain])]))
  5.0)


;; ------------------



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

(comment
  (->
   (recover
    item-memory
    (hdd/automaton-destination
     bread-domain
     (hdd/clj->vsa* {:bread {:surface :empty}})
     (hdd/clj->vsa* {:spread butter})))
   ;; asking what bread do I have after
   (hd/unbind
    (hdd/clj->vsa* :bread))
   (hd/unbind
    (hdd/clj->vsa* :surface))
   (hdd/cleanup*))
  '(:butter :liquid))

;; -----------------------------------------
;; Lets' say that the system has associated lava with liquid
;;

(def lava (hd/thin (hd/superposition liquid (hdd/clj->vsa :lava))))

;; hm, I can recover bread of course when I query with the supperposition of lava and butter
(hdd/cleanup*
 (hd/unbind (recover
             item-memory
             (hdd/automaton-source
              bread-domain
              (hdd/clj->vsa* {:bread {:surface
                                      #{lava butter}}})
              (hdd/clj->vsa* {:spread #{lava butter}})))
            (hdd/clj->vsa* {:surface :empty})))
'(:bread)

;; that's not remarkable
;;

(def bread-domain
  (hdd/finite-state-automaton-1
    (remember-leaves!
      (hdd/clj->vsa*
        [[#{:bread {:surface :empty}} {:spread butter}
          #{:bread {:surface butter}}]
         [#{:bread {:surface butter}} {:spread butter}
          #{:bread {:surface butter :thick? true}}]
         [#{:bread {:surface butter}} {:scrub-off butter}
          #{:bread {:surface butter :thin? true}}]]))))

(def rocks-domain
  (hdd/finite-state-automaton-1
    (remember-leaves!
      (hdd/clj->vsa*
       [[#{:rocks {:surface :empty}}
          {:spread lava}
          #{:rocks {:surface lava}}]
         [#{:rocks {:surface lava}} {:spread lava}
          #{:rocks {:surface lava :thick? true}}]
         [#{:rocks {:surface lava}}
          {:scrub-off lava}
          #{:rocks {:surface lava :thin? true}}]]))))

(hdd/cleanup*
 (recover item-memory
          (hdd/automaton-source
           bread-domain
           ;; note the lack of bread
           (hdd/clj->vsa* {:surface lava})
           (hdd/clj->vsa* {:spread lava}))))
'(:rocks :bread)


;; -------------------------
;; This works because {:spread lava} and {:spread butter} are similar
;;
;; This isn't much more remarkable that 'what is the dollar of mexico'
;; but uses a finite state automaton and ambiguity.
;;

;; If I query the intersection, I get rocks out:
(hdd/cleanup*
 (recover item-memory
          (hdd/automaton-source
           (hdd/intersection-1 [bread-domain
                              rocks-domain])
           ;; note the lack of bread
           (hdd/clj->vsa* {:surface lava})
           (hdd/clj->vsa* {:spread lava}))))
'(:rocks)

;; querying the intersection for butter yields bread
(hdd/cleanup*
 (recover item-memory
          (hdd/automaton-source
           (hdd/intersection-1 [bread-domain
                              rocks-domain])
           ;; note the lack of bread
           (hdd/clj->vsa* {:surface butter})
           (hdd/clj->vsa* {:spread butter}))))
'(:bread)


;;
;; set difference amplifies the difference betweent the domains
;;

;; these are non-deterministic outcomes, coinflips I guess
;; It depends on the seed vectors going in, so once the seedvectors are set, this is deterministic
;; (it also depends on what is in the item memory)
(hdd/cleanup* (recover
               item-memory
               (hdd/automaton-source
                (hdd/difference rocks-domain bread-domain)
                (hdd/clj->vsa* {:thin? true})
                (hdd/clj->vsa* {:scrub-off lava}))))
'(:rocks)

(hdd/cleanup* (recover
                item-memory
                (hdd/automaton-source
                  (hdd/difference rocks-domain bread-domain)
                  (hdd/clj->vsa* {:thin? true})
                  (hdd/clj->vsa* {:scrub-off butter}))))
'(:rocks)

;; (compare with intersection)
(hdd/cleanup*
 (recover
  item-memory
  (hdd/automaton-source
   (hdd/intersection-1 [rocks-domain bread-domain])
   (hdd/clj->vsa* {:thin? true})
   (hdd/clj->vsa* {:scrub-off butter}))))
'(:bread :rocks)

(hdd/cleanup*
 (recover
  item-memory
  (hdd/automaton-source
   (hdd/intersection-1 [rocks-domain bread-domain])
   (hdd/clj->vsa* {:thin? true})
   (hdd/clj->vsa* {:scrub-off #{butter lava}}))))
'(:bread :rocks)


;; with union
(hdd/cleanup*
 (recover
  item-memory
  (hdd/automaton-source
   (hdd/union rocks-domain bread-domain)
   (hdd/clj->vsa* {:thin? true})
   (hdd/clj->vsa* {:scrub-off butter}))))
'(:bread)
;; then bread, guess thatt is a coinflip with the item memory or sth in that case
;;

;; quering with both lava and butter
(hdd/cleanup*
 (recover
  item-memory
  (hdd/automaton-source
   (hdd/union rocks-domain bread-domain)
   (hdd/clj->vsa* {:thin? true})
   (hdd/clj->vsa* {:scrub-off #{butter lava}}))))
'(:rocks)
;; now rocks

;; rocks or nothing
(for [n (range 5)]
  (hdd/cleanup* (recover
                 item-memory
                 (hdd/automaton-source
                  (hdd/difference rocks-domain bread-domain)
                  (hdd/clj->vsa* {:thin? true})
                  (hdd/clj->vsa* {:scrub-off (hd/->seed)})))))
'((:rocks) (:rocks) (:rocks) (:rocks) ())


;; bread or nothing
;; guess that is random
(for [n (range 5)]
  (some->
   (recover
    item-memory
    (hdd/automaton-source
     ;; with intersection instead
     (hdd/intersection-1 [rocks-domain bread-domain])
     (hdd/clj->vsa* {:thin? true})
     (hdd/clj->vsa* {:scrub-off (hd/->seed)})))
   hdd/cleanup*))
'(nil (:bread) (:bread) nil nil)











;; -------------------------

(comment
  (remember-leaves!
   (hdd/clj->vsa*
    [[{:bread {:surface :empty}} {:spread butter}
      {:bread {:surface butter}}]
     [{:bread {:surface butter}} {:spread butter}
      {:bread {:surface butter :thick? true}}]
     [{:bread {:surface butter}} {:scrub-off butter}
      {:bread {:surface butter :thin? true}}]]))

  (hd/similarity
   (hdd/clj->vsa* {:bread {:surface :empty}})
   (recover item-memory (hdd/clj->vsa* {:bread {:surface butter}})))
  0.0
  (hd/similarity
   (hdd/clj->vsa* {:bread {:surface butter}})
   (recover item-memory (hdd/clj->vsa* {:bread {:surface butter}})))
  1.0)

(comment
  (do (remember item-memory
                (hdd/clj->vsa* #{:bread {:surface :empty}}))
      (= (hdd/clj->vsa* #{:bread {:surface :empty}})
         (recover item-memory
                  (hdd/clj->vsa* #{:bread
                                   {:surface :empty}}))))
  true
  (do
    (doseq [x (hdv-nodes
               (hdd/clj->vsa*
                [[#{:bread {:surface :empty}}
                  #{:spread :butter}
                  #{:bread {:surface :butter}}]
                 [#{:bread {:surface :butter}}
                  #{:spread :butter}
                  #{:bread
                    {:surface :butter :thick? true}}]
                 [#{:bread {:surface :butter}}
                  #{:scrub-off :butter}
                  #{:bread
                    {:surface :butter :thin? true}}]]))]
      (remember item-memory x)
      (remember item-memory
                (hdd/clj->vsa* #{{:surface :empty} :bread}))
      (remember item-memory
                (hdd/clj->vsa* #{{:surface :empty} :bread}))
      (remember item-memory
                (hdd/clj->vsa* #{{:surface :empty}
                                 :bread})))
    (let [bread1outcome (recover item-memory
                                 (hdd/clj->vsa*
                                  #{:bread
                                    {:surface :empty}}))]
      (map (fn [symbolic-item]
             {:bread1outcome-sum (f/sum bread1outcome)
              :equal? (= bread1outcome
                         (hdd/clj->vsa* symbolic-item))
              :hdv-sum (f/sum (hdd/clj->vsa* symbolic-item))
              :similarity (hd/similarity bread1outcome
                                         (hdd/clj->vsa*
                                          symbolic-item))
              :symbolic-item symbolic-item})
           [#{:bread {:surface :empty}} #{:spread :butter}
            #{:bread {:surface :butter}}
            #{:bread {:surface :butter}} #{:spread :butter}
            #{:bread {:surface :butter :thick? true}}
            #{:bread {:surface :butter}} #{:scrub-off :butter}
            #{:bread {:surface :butter :thin? true}}]))))
































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
