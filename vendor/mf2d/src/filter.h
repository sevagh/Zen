#ifndef FILTER_H
#define FILTER_H

// T: float or double.
// x, y: image dimensions, width x pixels and height y pixels.
// hx, hy: window radius in x and y directions.
// blockhint: preferred block size, 0 = pick automatically.
// in: input image.
// out: output image.
//
// Total window size will be (2*hx+1) * (2*hy+1) pixels.
//
// Pixel (i,j) for 0 <= i < x and 0 <= j < y is located at
// in[j*x + i] and out[j*x + i].

template <typename T>
void median_filter_2d(int x, int y, int hx, int hy, int blockhint, const T* in, T* out);

// As above, for the special case y = 1, hy = 0.

template <typename T>
void median_filter_1d(int x, int hx, int blockhint, const T* in, T* out);

#endif
