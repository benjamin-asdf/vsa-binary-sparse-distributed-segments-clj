(ns bennischwerdtner.hd.prot)

(defprotocol ItemMemory
  (m-clj->vsa [this item])
  (m-cleanup-verbose [this q]
    [this q threshold])
  (m-cleanup [this q])
  (m-cleanup* [this q]
    [this q threshold]))






(comment
  (clojure.string/replace)

  (let [specialVars (map second
                         (re-seq #"((\{(.+?)\})|(.+?))"
                                 "{name} foo {bar}"))])
  (let [input "hello {name} foo {bar} baz"
        regex #"(\{([^{}]+)\})|([^{}]+)"
        matches (re-seq regex input)]
    matches)
  '(["hello " nil nil "hello "]
    ["{name}" "{name}" "name" nil]
    [" foo " nil nil " foo "]
    ["{bar}" "{bar}" "bar" nil]
    [" baz" nil nil " baz"])
  (let [input "hello {name} foo{ {bar} baz"
        regex #"(\{([^{}]+)\})|([^{}]+)"
        blocks (re-seq regex input)]
    (into []
          (map (fn [[text _ specialVarContent
                     plainTextContent]]
                 (cond specialVarContent
                       (if-let [specialVar
                                (get {"bar" :bar
                                      "name" :name}
                                     specialVarContent)]
                         {:content specialVarContent
                          :specialVar specialVar}
                         ;;
                         {:content text :kind :plain})
                       plainTextContent {:content text
                                         :kind :plain}))
               blocks)))
  [{:content "hello " :kind :plain}
   {:content "name" :specialVar :name}
   {:content " foo" :kind :plain}
   {:content " " :kind :plain}
   {:content "bar" :specialVar :bar}
   {:content " baz" :kind :plain}])
