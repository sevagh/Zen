#!/usr/bin/env bash

#EXTRA_CXX_FLAGS=-DCMAKE_CXX_FLAGS="-fsanitize=undefined -fsanitize=address -lubsan"
#EXTRA_CXX_FLAGS=""

rm -rf build

mkdir -p build &&\
       	cd build &&\
	cmake ../libzengarden -G Ninja -DCMAKE_CXX_COMPILER=/home/sevagh/GCC-8.4.0/bin/g++ -DCMAKE_CUDA_HOST_COMPILER=/home/sevagh/GCC-8.4.0/bin/g++ -DIPP_ROOT=/home/sevagh/intel/ipp &&\
	ninja -j 16
	#ninja -j 16 &&\
	#CTEST_OUTPUT_ON_FAILURE=1 ninja test --label-exclude=".*_valgrind"
