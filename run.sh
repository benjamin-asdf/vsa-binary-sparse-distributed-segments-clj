#!/bin/sh

cd "$(dirname "$0")"

source ./activate.sh 

/usr/bin/clojure "$@"

