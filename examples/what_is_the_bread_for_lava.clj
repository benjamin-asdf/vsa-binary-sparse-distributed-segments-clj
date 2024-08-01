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
(def honey (hd/thin (hd/superposition liquid (hdd/clj->vsa :honey))))
(def butter (hd/thin (hd/superposition liquid (hdd/clj->vsa :butter))))
(def lava (hd/thin (hd/superposition liquid (hdd/clj->vsa :lava))))


;; -------------------------------------------------------
;; model as asssoicative map
;;
;; This is almost the same as dollar in mexico.
;; But now we we use the intersection as query.
;; This is programing in superposition and a little bit 'analogical', leveraging the intersection.
;;

(def bread-domain
  (hdd/clj->vsa* {:ground :bread :surface butter}))

(def rocks-domain
  (hdd/clj->vsa* {:ground :rocks :surface lava}))

(hdd/cleanup*
 (hd/unbind
  ;; ~ {:ground :rocks}
  (hd/unbind rocks-domain
             ;; intersection
             ;; ~ {:surface liquid}
             (hdd/intersection [bread-domain
                                rocks-domain]))
  (hdd/clj->vsa* :ground)))
'(:rocks)

;; to show that you can juggle these things a little,
;; let's query the union:

(hdd/cleanup*
 (hd/unbind
  ;; ~ {:ground (⊕ :rocks :bread)}
  (hd/unbind (hdd/union rocks-domain bread-domain)
             ;; intersection
             ;; ~ {:surface liquid}
             (hdd/intersection [bread-domain
                                rocks-domain]))
  (hdd/clj->vsa* :ground)))
'(:bread :rocks)


;; the difference is also meaningful
(hdd/cleanup*
 (hd/unbind
  (hd/unbind (hdd/difference rocks-domain bread-domain)
             ;; intersection
             ;; ~ {:surface liquid}
             (hdd/intersection [bread-domain
                                rocks-domain]))
  (hdd/clj->vsa* :ground)))
'(:rocks)

;; other way around:
(hdd/cleanup*
 (hd/unbind
  (hd/unbind (hdd/difference bread-domain rocks-domain)
             ;; intersection
             ;; ~ {:surface liquid}
             (hdd/intersection [bread-domain
                                rocks-domain]))
  (hdd/clj->vsa* :ground)))
'(:bread)





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
           (hdd/intersection [bread-domain
                              rocks-domain])
           ;; note the lack of bread
           (hdd/clj->vsa* {:surface lava})
           (hdd/clj->vsa* {:spread lava}))))
'(:rocks)

;; querying the intersection for butter yields bread
(hdd/cleanup*
 (recover item-memory
          (hdd/automaton-source
           (hdd/intersection [bread-domain
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
   (hdd/intersection [rocks-domain bread-domain])
   (hdd/clj->vsa* {:thin? true})
   (hdd/clj->vsa* {:scrub-off butter}))))
'(:bread :rocks)

(hdd/cleanup*
 (recover
  item-memory
  (hdd/automaton-source
   (hdd/intersection [rocks-domain bread-domain])
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
     (hdd/intersection [rocks-domain bread-domain])
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
