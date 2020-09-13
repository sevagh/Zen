# zengarden

Zengarden aims to become a realtime-capable GPU-accelerated rhythm analysis toolkit written in C++. It consists of:
* libzengarden, a C++ library that depends on CUDA and IPP
* zengarden, a reference command-line client, also written in C++

libzengarden is designed from the ground up to support dual CPU/GPU implementations of algorithms by using policy-based template metaprogramming. For specialized subroutines (e.g. cuFFT, Npp/Ipp), there are abstraction wrappers.

### Origin

Having originally worked on [Real-time Harmonic-Percussive Source Separation](https://github.com/sevagh/Real-Time-HPSS), my conclusion at the time was that although the technique was viable, the running time (5ms for 10ms-sized inputs) made it unsuitable for true real-time applications.

Believing that harmonic-percussive separation could be an important preprocessing step to improving the accuracy of existing beat and downbeat-tracking algorithms, I tried to write an optimized version using CUDA. By using the NppiFilterMedian family of functions for implementing median-filtering-based HPS, I got the computation time down to ~160us for a 10ms input buffer.

When processing is done on a GPU, the data should stay on the GPU - that's why expanding the HPS algorithm into a library to house a collection of beat tracking algorithms would help minimize the cost of adding HPS as a preprocessing step, since the cost of shipping single buffers of input at a time is amortized across all the processing we can do on it.

### Roadmap

The initial roadmap of zengarden is focused on pre-processing:

1. Add a variant of HPS from https://iie.fing.edu.uy/publicaciones/2014/Iri14/Iri14.pdf
2. Write a test suite that evaluates existing state-of-the-art beat trackers (using MIREX challenge datasets, etc.), to demonstrate the usefulness of HPS as a preprocessing step in beat-tracking algorithms
3. Look into dynamic range processing - specifically expanders, noisegates, or transient shapers - to potentially chain with HPS to create an even better separation of percussive audio

Once I can (hopefully) demonstrate that HPS significant boosts general beat tracking accuracy, further direction will be decided.

### Development

I currently build and compile zengarden on Linux (Fedora 32) using GCC 8, CUDA Toolkit 10.2, and nvcc on an amd64 Ryzen host (hence the name **zen**garden) with an NVIDIA RTX 2070 SUPER. All NVIDIA tools are installed using negativo17's nvidia repository.

#### Tests 

There are unit tests in the libzengarden source tree. Memory and UB checks can be run during the test suite as follows. I favor asan over valgrind, but we need some special ASAN options to not clash with CUDA. I also try to use cuda-memcheck, but it slows execution down too much in some cases.

```
$ mkdir -p build && cd build && cmake .. -GNinja -DENABLE_UBSAN=ON -DENABLE_ASAN=ON
$ ninja -j16
$ export ASAN_OPTIONS="protect_shadow_gap=0:replace_intrin=0:detect_leaks=0"
$ ninja test
```
