#!/usr/bin/env bash

UBSAN="-DENABLE_UBSAN=ON"
ASAN="-DENABLE_ASAN=ON"
export ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0

rm -rf build

mkdir -p build &&\
       	cd build &&\
	cmake ../libzengarden -G Ninja -DCMAKE_CXX_COMPILER=/home/sevagh/GCC-8.4.0/bin/g++ -DCMAKE_CUDA_HOST_COMPILER=/home/sevagh/GCC-8.4.0/bin/g++ -DIPP_ROOT=/home/sevagh/intel/ipp ${UBSAN} ${ASAN} &&\
	ninja -j 16 &&\
	CTEST_OUTPUT_ON_FAILURE=1 ninja test

ninja cpp-clean || true
