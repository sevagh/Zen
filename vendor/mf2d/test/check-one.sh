#!/bin/bash

time test/run-one.sh "$1" || exit 1
echo
if [ "$(uname)" == "Darwin" ]; then
    md5 tmp/*.fits > "tmp/checksum" || exit 1
    cmp "test/checksum-md5" "tmp/checksum" || exit 1
else
    md5sum tmp/*.fits > "tmp/checksum" || exit 1
    cmp "test/checksum-md5sum" "tmp/checksum" || exit 1
fi

