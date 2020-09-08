#!/usr/bin/env bash

BUILD_DIR="build-cli"

rm -rf $BUILD_DIR

mkdir -p ${BUILD_DIR} &&\
       	cd ${BUILD_DIR} &&\
	cmake ../zengarden -G Ninja -DCMAKE_CXX_COMPILER=/home/sevagh/GCC-8.4.0/bin/g++ -DCMAKE_CUDA_HOST_COMPILER=/home/sevagh/GCC-8.4.0/bin/g++ -DIPP_ROOT=/home/sevagh/intel/ipp &&\
	ninja -j 16
