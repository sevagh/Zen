#!/bin/bash

echo -n "unit test: "
"$1/mf2d-unittest" || exit 1
for i in 0 1 2 3 5 10 100 1000; do
    echo -n .
    "$1/mf2d" "$i" "example/test-0.fits" "!tmp/med-0-$i.fits" "!tmp/dif-0-$i.fits" || exit 1
done
for i in 0 1 2 3 5 10 15 25 50; do
    echo -n :
    "$1/mf2d" "$i" "example/test-1.fits" "!tmp/med-1-$i.fits" "!tmp/dif-1-$i.fits" || exit 1
done
echo
