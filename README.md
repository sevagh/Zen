# Zen

Zen is a real-time CUDA-accelerated harmonic/percussive (or steady-state/transient) source separation library. More specifically, it implements:
* Harmonic-percussive source separation using median filtering ([Fitzgerald 2010](http://dafx10.iem.at/papers/DerryFitzGerald_DAFx10_P15.pdf), [Drieger et al 2014](https://archives.ismir.net/ismir2014/paper/000127.pdf))
* Steady-state/transient source separation using SSE (stochastic spectrum estimation) filtering ([Bayarres 2014](https://iie.fing.edu.uy/publicaciones/2014/Iri14/Iri14.pdf))

Zen was written from the ground up to support dual CPU/GPU implementations of algorithms by using policy-based template metaprogramming. For specialized subroutines (e.g. cuFFT, Npp/Ipp), there are abstraction wrappers.

## Block diagrams

## Results

### Performance

### Accuracy improvements

#### Beat tracking (BTrack)

#### Pitch tracking (McLeod Pitch Method)

## Examples

## Origin

This is a followup to my project [Real-time Harmonic-Percussive Source Separation](https://github.com/sevagh/Real-Time-HPSS). In the previous project, I showed that Fitzgerald's 2010 algorithm for median-filtering harmonic-percussive source separation (and Drieger et al's subequent 2014 modification) could be adapted to work in real-time. However, my simple MATLAB and Python implementations were too slow to be feasible (~5-10ms of processing per 10ms hop in a real-time stream).

Using CUDA and NPP to implement median-filtering-based HPR (harmonic-percussive-residual) separation, I got the computation time down to ~160us for a 10ms input buffer in this library, making it viable as an early stage in a real-time processing chain. The name is Zen because I wrote it on a Ry**zen**-based computer and tested it on Meshuggah's album Ob**zen**.

## Usefulness

Harmonic separation in real-time is worse than offline. This is due to the large hop size (4096 samples, 85ms @ 48kHz) required for good harmonic separation. However, a small hop size (256/512 samples, 5-10ms) is suitable for percussive separation. Therefore, one could (or should) use this library as a real-time pre-processing step before applying percussion analysis algorithms (e.g. beat tracking, tempo tracking). Also note that code for better quality (but lower performance) offline variants of the algorithms are also included.

### Usage

Zen consists of:
* libzen, a C++ library that depends on CUDA and IPP
* zen, a reference command-line client, also written in C++

#### libzen library examples

#### zen command-line tool usage

## Development

I currently build and compile zengarden on Linux (Fedora 32) using GCC 8, CUDA Toolkit 10.2, and nvcc on an amd64 Ryzen host with an NVIDIA RTX 2070 SUPER. All NVIDIA libraries were installed and managed using negativo17's Fedora nvidia repository.

There are unit tests in the libzengarden source tree. Memory and UB checks can be run during the test suite as follows. I favor asan over valgrind, but we need some special ASAN options to not clash with CUDA. I also try to use cuda-memcheck, but it slows execution down too much in some cases.

```
$ mkdir -p build && cd build && cmake .. -GNinja -DENABLE_UBSAN=ON -DENABLE_ASAN=ON
$ ninja -j16
$ export ASAN_OPTIONS="protect_shadow_gap=0:replace_intrin=0:detect_leaks=0"
$ ninja test
```
