#!/bin/bash

export LC_ALL=C
rm -rf tmp
mkdir -p tmp || exit 1
for a in "$@"; do
    echo "$a"
    test/check-one.sh "$a" || exit 1
done
