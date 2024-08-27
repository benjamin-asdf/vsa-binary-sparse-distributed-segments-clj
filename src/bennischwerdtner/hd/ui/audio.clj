;;
;; Probably only works on linux with ffplay installed
;;
(ns bennischwerdtner.hd.ui.audio
  (:require
   [bennischwerdtner.hd.binary-sparse-segmented :as hd]
   [clojure.java.shell]))

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

(comment
  (play! (->audio {:frequency 440 :duration 0.2}))
  (play! (->audio {:frequency 1200 :duration 0.2}))
  (play! (->audio {:frequency 80 :duration 0.2}))
  (play! (->audio {:frequency 250 :duration 0.2})))

(defn render-hd-frequency
  [hd]
  (+ 250
     (* (- 1100 250)
        ;; between 0 and 500
        (/ (first (hd/hv->indices hd))
           (:bsdc-seg/segment-length hd/default-opts)))))

(defn render-hd-duration
  [hd]
  (+ 0.05 (* 0.3 (/ (first (hd/hv->indices hd)) 500))))

(defn render-hd-audio
  [hd]
  (->audio {:frequency (render-hd-frequency hd)
            :duration
              ;; (render-hd-duration hd)
            0.1}))

(comment
  (play! (->audio {:duration 0.1
                   :frequency (render-hd-frequency
                               (hd/->seed))})))

(defn listen!
  [hdvs]
  (future (doseq [hdv hdvs] (play! (render-hd-audio hdv)))))

(defn listen-seqs!
  [seq]
  (doseq [x seq]
    (listen! x)
    ;; (Thread/sleep (rand-nth [0 25 50]))
    ))

(comment
  (def alphabet (into [] (repeatedly 24 hd/->seed)))
  ;; 'abc' + * 'xyz'
  (listen-seqs! [(take 3 alphabet)
                 (map #(hd/bind %1 %2)
                      (take 3 alphabet)
                      (take-last 3 alphabet))])

  (listen-seqs!
   [(take 3 alphabet)
    (map #(hd/bind %1 %2)
         (take 3 alphabet)
         (take-last 3 alphabet))
    ;; (take-last 3 alphabet)
    ])

  ;; 'abc'
  (listen-seqs! [ ;; (take 3 alphabet)
                 ;; (take 6 alphabet)
                 (take 7 alphabet)])

  (listen-seqs! [(take 7 (drop 7 alphabet))])

  (listen-seqs! [(take (* 7 2) alphabet)])



  ;; they way these have 'character' is so mesmerizing to me
  )
