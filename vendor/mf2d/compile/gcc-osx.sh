#!/bin/bash

# If all else fails, try to remove both "-Wa,-q" and "-march=native".

set -x
mkdir -p bin || exit 1
cd src || exit 1
${CXX:-g++} -Wall -march=native -O3 -fopenmp -Wa,-q -DNDEBUG mf2d.cc driver.cc filter.cc imageio.cc -lcfitsio -o ../bin/mf2d || exit 1
${CXX:-g++} -Wall -march=native -O3 -fopenmp -Wa,-q -DNDEBUG mf2d-benchmark.cc driver.cc filter.cc imageio.cc -lcfitsio -o ../bin/mf2d-benchmark || exit 1
${CXX:-g++} -Wall -march=native -O3 -fopenmp -Wa,-q -DNDEBUG mf2d-unittest.cc filter.cc -o ../bin/mf2d-unittest || exit 1
