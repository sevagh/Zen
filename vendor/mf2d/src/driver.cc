#include <cmath>
#include <iostream>
#include <stdexcept>
#include "driver.h"
#include "filter.h"
#include "imageio.h"
#include "timer.h"

//// Parameters for 1D and 2D filtering

struct Param {
    int max_h;
    double bf0;
    double bf1;
};
const Param param1d = {1000, 5.0, 11.0};
const Param param2d = {100, 3.0, 6.0};

const int BSTEPS = 6;

static double get_factor(const Param* param, int i) {
    double ifract = static_cast<double>(i) / BSTEPS;
    return param->bf0 + (param->bf1 - param->bf0) * ifract;
}

static int benchmark_step(int h) {
    if (h < 10) {
        return h + 1;
    } else if (h < 100) {
        return h + 10;
    } else {
        return h + 100;
    }
}

template <typename T>
static const Param* get_param(const Driver<T,Image1D<T> > *tag) {
    return &param1d;
}

template <typename T>
static const Param* get_param(const Driver<T,Image2D<T> > *tag) {
    return &param2d;
}

//// Helper functions

template <typename T>
static void filter(int h, int blockhint, Image1D<T> in, Image1D<T> out) {
    median_filter_1d<T>(in.x, h, blockhint, in.p, out.p);
}

template <typename T>
static void filter(int h, int blockhint, Image2D<T> in, Image2D<T> out) {
    median_filter_2d<T>(in.x, in.y, h, h, blockhint, in.p, out.p);
}

template <typename T>
static void compare(T** prev, const T* cur, int size) {
    if (*prev) {
        for (int i = 0; i < size; ++i) {
            T a = cur[i];
            T b = (*prev)[i];
            bool ok = (a == b) || (std::isnan(a) && std::isnan(b));
            if (!ok) {
                throw std::runtime_error("output mismatch");
            }
        }
    } else {
        *prev = new T[size];
        for (int i = 0; i < size; ++i) {
            (*prev)[i] = cur[i];
        }
    }
}

//// Method implementations

template <typename T, typename I>
void Driver<T,I>::process(int h) {
    filter<T>(h, 0, in, out);
}

template <typename T, typename I>
void Driver<T,I>::diff() {
    for (int i = 0; i < out.size(); ++i) {
        out.p[i] = in.p[i] - out.p[i];
    }
}

template <typename T, typename I>
void Driver<T,I>::write(const char* filename) {
    write_image(filename, out);
}

template <typename T, typename I>
void Driver<T,I>::benchmark() {
    const Param* param = get_param(this);
    for (int i = 0; i <= BSTEPS; ++i) {
        double bfactor = get_factor(param, i);
        std::cout << "\t" << bfactor;
    }
    std::cout << std::endl;
    for (int h = 0; h <= param->max_h; h = benchmark_step(h)) {
        std::cout << h << std::flush;
        T* prev = 0;
        for (int i = 0; i <= BSTEPS; ++i) {
            double bfactor = get_factor(param, i);
            int blocksize = static_cast<int>(bfactor * (h + 2));
            Timer timer;
            filter<T>(h, blocksize, in, out);
            double t = timer.peek();
            std::cout << "\t" << t << std::flush;
            compare<T>(&prev, out.p, out.size());
        }
        if (prev) {
            delete[] prev;
        }
        std::cout << std::endl;
    }
}

//// Versions

template class Driver<float, Image1D<float> >;
template class Driver<float, Image2D<float> >;
template class Driver<double, Image1D<double> >;
template class Driver<double, Image2D<double> >;
