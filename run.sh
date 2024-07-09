#!/bin/sh

source ./activate.sh || exit 10

/usr/bin/clojure "$@"
