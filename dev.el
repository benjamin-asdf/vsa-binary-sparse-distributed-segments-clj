(let ((cider-clojure-cli-command (expand-file-name "./run.sh")))
  (cider-jack-in-clj
   '(:project-type clojure-cli)))
