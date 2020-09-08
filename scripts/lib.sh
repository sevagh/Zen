#!/usr/bin/env bash

#EXTRA_CXX_FLAGS=-DCMAKE_CXX_FLAGS="-fsanitize=undefined -fsanitize=address -lubsan"
#EXTRA_CXX_FLAGS=""

BUILD_DIR="build-lib"

rm -rf ${BUILD_DIR}

mkdir -p ${BUILD_DIR} &&\
       	cd ${BUILD_DIR} &&\
	cmake ../libzengarden -G Ninja -DCMAKE_CXX_COMPILER=/home/sevagh/GCC-8.4.0/bin/g++ -DCMAKE_CUDA_HOST_COMPILER=/home/sevagh/GCC-8.4.0/bin/g++ -DIPP_ROOT=/home/sevagh/intel/ipp -DBUILD_BENCHES=ON -DBUILD_TESTS=OFF &&\
	ninja -j 16 &&\
	sudo ninja install
