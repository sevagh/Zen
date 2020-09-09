#!/usr/bin/env bash

set -euxo pipefail

BUILDCMD=${1:-"build"}
BUILD_DIR_PARENT="./build"

if [ "${BUILDCMD}" == "build" ]; then
	mkdir -p ${BUILD_DIR_PARENT}

	echo "working in ${BUILD_DIR_PARENT}"

	BUILD_DIR="${BUILD_DIR_PARENT}/build-lib"
	echo "building libzengarden into ${BUILD_DIR}"

	mkdir -p ${BUILD_DIR} &&\
		cd ${BUILD_DIR} &&\
		cmake ../../libzengarden -G Ninja -DCMAKE_CXX_COMPILER=/home/sevagh/GCC-8.4.0/bin/g++ -DCMAKE_CUDA_HOST_COMPILER=/home/sevagh/GCC-8.4.0/bin/g++ -DIPP_ROOT=/home/sevagh/intel/ipp -DBUILD_TESTS=OFF &&\
		ninja -j 16

	cd -

	BUILD_DIR="${BUILD_DIR_PARENT}/build-cli"
	echo "building zengarden cli into ${BUILD_DIR}"

	mkdir -p ${BUILD_DIR} &&\
		cd ${BUILD_DIR} &&\
		cmake ../../zengarden -G Ninja -DCMAKE_CXX_COMPILER=/home/sevagh/GCC-8.4.0/bin/g++ -DCMAKE_CUDA_HOST_COMPILER=/home/sevagh/GCC-8.4.0/bin/g++ -DIPP_ROOT=/home/sevagh/intel/ipp &&\
		ninja -j 16
elif [ "${BUILDCMD}" == "clean" ]; then
	rm -r "${BUILD_DIR_PARENT}"
elif [ "${BUILDCMD}" == "test" ]; then
	UBSAN="-DENABLE_UBSAN=ON"
	ASAN="-DENABLE_ASAN=ON"
	export ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0

	BUILD_DIR="${BUILD_DIR_PARENT}/build-test"

	mkdir -p ${BUILD_DIR} &&\
		cd ${BUILD_DIR} &&\
		cmake ../../libzengarden -G Ninja -DCMAKE_CXX_COMPILER=/home/sevagh/GCC-8.4.0/bin/g++ -DCMAKE_CUDA_HOST_COMPILER=/home/sevagh/GCC-8.4.0/bin/g++ -DIPP_ROOT=/home/sevagh/intel/ipp ${UBSAN} ${ASAN} &&\
		ninja -j 16 &&\
		CTEST_OUTPUT_ON_FAILURE=1 ninja test

	ninja cpp-clean || true
elif [ "${BUILDCMD}" == "fmt" ]; then
	echo "running clang-format on all source files"
	find libzengarden/ zengarden/ ZenGardenia/ -regex '.*\.\(cu\|h\)' -exec clang-format -style=file -i {} \;
fi
