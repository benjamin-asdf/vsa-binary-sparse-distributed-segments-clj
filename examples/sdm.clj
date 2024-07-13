(ns sdm
  (:require
   [bennischwerdtner.hd.binary-sparse-segmented :as hd]
   [tech.v3.datatype :as dtype]
   [tech.v3.tensor :as dtt]
   [tech.v3.parallel.for :as pfor]
   [tech.v3.datatype.argops :as dtype-argops]
   [tech.v3.datatype.functional :as f]
   [clojure.math.combinatorics :as combo]
   [bennischwerdtner.sdm.sdm :as sdm]))

(defprotocol AutoAssociativeMemory
  (lookup [this address]
    [this address top-k])
  (store [this address])
  (known? [this address]))

(defn ->auto-a-memory
  []
  (let [address-count (long (* 1.5 1e4))
        word-length (:bsdc-seg/N hd/default-opts)
        address-density 0.0009
        ;; 0.0009
        ;; (long (/ 6 address-count))
        decoder-threshold 2
        read-threshold 8
        state (atom {:content-matrix (sdm/->content-matrix
                                       address-count
                                       word-length)
                     :decoder
                       (sdm/->address-decoder
                         {:address-count address-count
                          :address-density address-density
                          :word-length word-length})})]
    (reify
      AutoAssociativeMemory
        (lookup [this address k]
          (let [{:keys [decoder content-matrix]} @state]
            (-> (sdm/lookup-iteratively
                  content-matrix
                  address
                  decoder
                  (merge hd/default-opts
                         {:decoder-threshold
                            decoder-threshold
                          :read-threshold read-threshold
                          :top-k 1}))
                :result)))
        (lookup [this address] (lookup this address 1))
        (store [this address]
          (let [{:keys [decoder content-matrix]} @state]
            (sdm/auto-associate! content-matrix
                                 address
                                 decoder
                                 decoder-threshold))))))


;; (defonce auto-a-memory (->auto-a-memory))

(comment


  ;; Question:
  ;; I approach the memory with a noisy version of a stored vector `a`,
  ;; how much noise does it tolerate to get `a` out,
  ;; at the cost of returning `a` for an unrelated query vector `b`

  (def activation-probability-roughly 0.003801)
  (def address-count (long 1e4))


  (apply max
         (let [a (hd/->seed)]
           (for [t (map (fn [_] (hd/->seed)) (range 1e5))]
             (hd/similarity t a))))
  0.08


  ;; generally, 0.09 similarity should suffice to say this is 'similar'
  ;;

  (* activation-probability-roughly address-count 0.09)
  3.4208999999999996

  ;; then, activation threshold would be something like 3.4
  ;;

  ;;
  (/ 7 (* activation-probability-roughly address-count))
  0.1841620626151013
  ;; I thought this would mean with threshold 7, similarity 0.18 should roughly be
  ;; sufficient
  ;; maybe it's the mean probability and the std deviation is high
  ;;

  (* 7.825E-4 1e6)

  )


(comment
  (def auto-a-memory (->auto-a-memory))
  (def auto-a-memory nil)


  (def T (into [] (repeatedly 1e3 #(hd/->seed))))
  nil

  (time
   (doseq [t T]
     (store auto-a-memory t)))

  ;; question: Can I approach the memeory with 0.2 similar vector and get my stored vector out?
  ;; with threshold 7
  ;;

  ;; doesn't work:
  (doall
   (for [t (take 10 T)]
     (let [q (hd/weaken t 0.8)] (lookup auto-a-memory q))))
  (nil nil nil nil nil nil nil nil nil nil)

  ;; didn't turn out to work like I thought
  (doall
   (for [t (take 10 T)]
     (let [q (hd/weaken t 0.2)] (lookup auto-a-memory q)))))



(comment
  (def auto-a-memory (->auto-a-memory))
  (def auto-a-memory nil)


  (def T (into [] (repeatedly 1e3 #(hd/->seed))))
  nil

  (time
   (doseq [t T]
     (store auto-a-memory t)))

  ;; question: Can I approach the memeory with 0.2 similar vector and get my stored vector out?
  ;; with threshold 5
  ;;


  (doall
   (for [t (take 10 T)]
     (let [q (hd/weaken t 0.8)] (lookup auto-a-memory q))))

  (nil nil nil nil nil nil nil nil nil nil)

  ;; didn't turn out to work like I thought
  (doall
   (for [t (take 10 T)]
     (let [q (hd/weaken t 0.2)]
       (hd/similarity t (lookup auto-a-memory q)))))
  '(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0)

  ;; 0.6, some start to fail:
  (doall
   (for [t (take 10 T)]
     (let [q (hd/weaken t 0.6)] (lookup auto-a-memory q)))))



(comment
  (def auto-a-memory (->auto-a-memory))
  (def auto-a-memory nil)


  (def T (into [] (repeatedly 1e3 #(hd/->seed))))
  nil

  (time
   (doseq [t T]
     (store auto-a-memory t)))

  ;; question: Can I approach the memeory with 0.2 similar vector and get my stored vector out?
  ;; with threshold 4
  ;;

  ;; with 4,

  (doall
   (for [t (take 10 T)]
     (let [q (hd/weaken t 0.8)]
       (some->
        (lookup auto-a-memory q)
        (hd/similarity t)))))
  '(1.0 nil nil nil nil nil 1.0 nil nil nil)


  ;; didn't turn out to work like I thought
  (doall
   (for [t (take 10 T)]
     (let [q (hd/weaken t 0.2)]
       (hd/similarity t (lookup auto-a-memory q)))))
  '(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0)

  ;; 0.6, some start to fail:
  (doall
   (for [t (take 10 T)]
     (let [q (hd/weaken t 0.6)] (lookup auto-a-memory q))))
  ;; still the case
  )



(comment
  (def auto-a-memory (->auto-a-memory))
  (def auto-a-memory nil)


  (def T (into [] (repeatedly 1e3 #(hd/->seed))))
  nil

  (time
   (doseq [t T]
     (store auto-a-memory t)))

  ;; question: Can I approach the memeory with 0.2 similar vector and get my stored vector out?
  ;; with threshold 3
  ;;

  (doall
   (for [t (take 10 T)]
     (let [q (hd/weaken t 0.8)]
       (some->
        (lookup auto-a-memory q)
        (hd/similarity t)))))
  '(1.0 nil nil 1.0 nil nil nil nil 1.0 nil)



  ;; didn't turn out to work like I thought
  (doall
   (for [t (take 10 T)]
     (let [q (hd/weaken t 0.2)]
       (hd/similarity t (lookup auto-a-memory q)))))
  '(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0)

  ;; 0.6, some start to fail:
  ;; with threshold 3, at 0.7 some start to fail:
  (doall
   (for [t (take 10 T)]
     (let [q (hd/weaken t 0.7)]
       (some-> (lookup auto-a-memory q)
               (hd/similarity t)))))

  (doall
   (for [t (take 10 T)]
     (let [q (hd/thin (hd/bundle t
                                 (hd/->seed)
                                 (hd/->seed)
                                 (hd/->seed)))]
       (some-> (lookup auto-a-memory q)
               (hd/similarity t))))))


(comment
  (def auto-a-memory (->auto-a-memory))
  (def auto-a-memory nil)


  (def T (into [] (repeatedly 1e3 #(hd/->seed))))
  nil

  (time
   (doseq [t T]
     (store auto-a-memory t)))

  ;; question: Can I approach the memeory with 0.2 similar vector and get my stored vector out?
  ;; address-count (long (* 1.5 1e4))
  ;; read-threshold 10
  ;;

  (doall
   (for [t (take 10 T)]
     (let [q (hd/weaken t 0.8)]
       (some->
        (lookup auto-a-memory q)
        (hd/similarity t)))))

  '(1.0 nil nil 1.0 nil nil nil nil 1.0 nil)


  ;; didn't turn out to work like I thought
  (doall (for [t (take 10 T)]
           (let [q (hd/weaken t 0.2)]
             (some-> (lookup auto-a-memory q)
                     (hd/similarity t)))))

  ;; sharp dropoff at 0.6
  (doall (for [t (take 10 T)]
           (let [q (hd/weaken t 0.6)]
             (some-> (lookup auto-a-memory q)
                     (hd/similarity t)))))

  (doall
   (for [t (take 10 T)]
     (let [q (hd/thin (hd/bundle t
                                 (hd/->seed)
                                 (hd/->seed)
                                 (hd/->seed)))]
       (some-> (lookup auto-a-memory q)
               (hd/similarity t))))))


(comment
  (def auto-a-memory (->auto-a-memory))
  (def auto-a-memory nil)


  (def T (into [] (repeatedly 1e3 #(hd/->seed))))
  nil

  (time
   (doseq [t T]
     (store auto-a-memory t)))

  ;; question: Can I approach the memeory with 0.2 similar vector and get my stored vector out?
  ;; address-count (long (* 1.5 1e4))
  ;; read-threshold 7
  ;;

  (doall
   (for [t (take 10 T)]
     (let [q (hd/weaken t 0.8)]
       (some->
        (lookup auto-a-memory q)
        (hd/similarity t)))))

  '(1.0 nil nil 1.0 nil nil nil nil 1.0 nil)

  (doall (for [t (take 100 T)]
           (let [q (hd/weaken t 0.5)]
             (some-> (lookup auto-a-memory q)
                     (hd/similarity t)))))


  ;; sharp dropoff at 0.6
  (doall
   (for [t (take 100 T)]
     (let [q (hd/weaken t 0.6)]
       (some-> (lookup auto-a-memory q)
               (hd/similarity t)))))


  (doall
   (for [t (take 10 T)]
     (let [q (hd/thin (hd/bundle t
                                 (hd/->seed)
                                 (hd/->seed)
                                 (hd/->seed)))]
       (some-> (lookup auto-a-memory q)
               (hd/similarity t)))))

  (1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.01 1.0 1.0))





(comment
  (def auto-a-memory (->auto-a-memory))
  (def auto-a-memory nil)
  (def T (into [] (repeatedly 1e3 #(hd/->seed))))
  nil
  (time (doseq [t T] (store auto-a-memory t)))
  ;; ... you can store what you care about twice
  ;; question: Can I approach the memeory with 0.2
  ;; similar vector and get my stored vector out?
  ;; address-count (long (* 1.5 1e4))
  ;; read-threshold 8
  ;; stored twice
  ;;
  (doall (for [t (take 10 T)]
           (let [q (hd/weaken t 0.8)]
             (some-> (lookup auto-a-memory q)
                     (hd/similarity t)))))
  '(1.0 nil nil nil nil nil nil nil nil 1.0)
  (filter nil?
          (doall (for [t T]
                   (let [q (hd/weaken t 0.5)]
                     (some-> (lookup auto-a-memory q)
                             (hd/similarity t))))))
  '()
  ;; sharp dropoff at 0.6
  (filter nil?
          (doall (for [t (take 100 T)]
                   (let [q (hd/weaken t 0.6)]
                     (some-> (lookup auto-a-memory q)
                             (hd/similarity t))))))
  '(nil nil nil nil)
  (doall (for [t (take 10 T)]
           (let [q (hd/thin (hd/bundle t
                                       (hd/->seed)
                                       (hd/->seed)
                                       (hd/->seed)))]
             (some-> (lookup auto-a-memory q)
                     (hd/similarity t))))))

;; I learn from this that representing what you care about to yourself works in this kind of system.
;; Sheds light on sharp wave ripples
;; One can postulate for cerebellum that there would be mechanisms that activate the same mossy + climbing fibers
;; multiple times, that this would happen during learning when the system decides a piece of data is important
;;
;;
