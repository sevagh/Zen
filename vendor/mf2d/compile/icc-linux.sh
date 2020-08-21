#!/bin/bash

set -x
mkdir -p bin || exit 1
cd src || exit 1
${CXX:-icpc} -Wall -O3 -xHost -fopenmp -DNDEBUG mf2d.cc driver.cc filter.cc imageio.cc -lcfitsio -o ../bin/mf2d || exit 1
${CXX:-icpc} -Wall -O3 -xHost -fopenmp -DNDEBUG mf2d-benchmark.cc driver.cc filter.cc imageio.cc -lcfitsio -o ../bin/mf2d-benchmark || exit 1
${CXX:-icpc} -Wall -O3 -xHost -fopenmp -DNDEBUG mf2d-unittest.cc filter.cc -o ../bin/mf2d-unittest || exit 1
