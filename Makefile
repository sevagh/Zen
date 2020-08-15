EXTRA_CMAKE_FLAGS:=-DCMAKE_CXX_COMPILER=/home/sevagh/GCC-8.4.0/bin/g++ -DCMAKE_CUDA_HOST_COMPILER=/home/sevagh/GCC-8.4.0/bin/g++

all: build

clean:
	rm -rf build/
	rm -rf lib/
	rm -rf bin/

setup:
	mkdir -p build
	(cd build; cmake .. -G Ninja $(EXTRA_CMAKE_FLAGS))

build: setup
	ninja -C build

clang-format: setup
	ninja -C build clang-format

.PHONY:
	clean
