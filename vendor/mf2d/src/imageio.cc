#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include "fitsio.h"
#include "imageio.h"


template <typename T> const char* get_type_descr();
template <> const char* get_type_descr<float>() { return "32-bit floats"; }
template <> const char* get_type_descr<double>() { return "64-bit floats"; }

template <typename T> int get_fits_type();
template <> int get_fits_type<float>() { return TFLOAT; }
template <> int get_fits_type<double>() { return TDOUBLE; }

template <typename T> int get_fits_bitpix();
template <> int get_fits_bitpix<float>() { return FLOAT_IMG; }
template <> int get_fits_bitpix<double>() { return DOUBLE_IMG; }

static void fcheck(int s) {
    if (s) {
        fits_report_error(stderr, s);
        std::exit(EXIT_FAILURE);
    }
}


template <typename T>
static void read_image_data(fitsfile* f, T* data, int size) {
    int s = 0;
    T nulval = std::numeric_limits<T>::quiet_NaN();
    int anynul = 0;
    fcheck(fits_read_img(f, get_fits_type<T>(), 1, size, &nulval, data, &anynul, &s));
    fcheck(fits_close_file(f, &s));
}


template <typename T>
static Image2D<T> read_image_data_2d(fitsfile* f) {
    int s = 0;
    long naxes[2] = {0,0};
    fcheck(fits_get_img_size(f, 2, naxes, &s));
    int64_t x = naxes[0];
    int64_t y = naxes[1];
    if (x < 1 || y < 1) {
        std::cerr << "image dimensions "
            << x << "x" << y << " too small" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (x * y >= std::numeric_limits<int>::max()) {
        std::cerr << "image dimensions "
            << x << "x" << y << " too large" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    Image2D<T> img(static_cast<int>(x), static_cast<int>(y));
    img.alloc();
    read_image_data<T>(f, img.p, img.size());
    return img;
}


template <typename T>
static Image1D<T> read_image_data_1d(fitsfile* f) {
    int s = 0;
    long naxes[1] = {0};
    fcheck(fits_get_img_size(f, 1, naxes, &s));
    long x = naxes[0];
    if (x < 1) {
        std::cerr << "image dimension "
            << x << " too small" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (x >= std::numeric_limits<int>::max()) {
        std::cerr << "image dimension "
            << x << " too large" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    Image1D<T> img(static_cast<int>(x));
    img.alloc();
    read_image_data<T>(f, img.p, img.size());
    return img;
}


static fitsfile* open_image_for_reading(const char* filename) {
    fitsfile* f = NULL;
    int s = 0;
    fcheck(fits_open_file(&f, filename, READONLY, &s));
    int hdutype = 0;
    fcheck(fits_get_hdu_type(f, &hdutype, &s));
    if (hdutype != IMAGE_HDU) {
        std::cerr << "expected IMAGE_HDU, got ";
        if (hdutype == ASCII_TBL) {
            std::cerr << "ASCII_TBL";
        } else if (hdutype == BINARY_TBL) {
            std::cerr << "BINARY_TBL";
        } else {
            std::cerr << hdutype;
        }
        std::cerr << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return f;
}


template <typename T>
static VDriver* from_image_helper(fitsfile* f) {
    int s = 0;
    int naxis = 0;
    fcheck(fits_get_img_dim(f, &naxis, &s));
    if (naxis == 1) {
        Image1D<T> img = read_image_data_1d<T>(f);
        return new Driver<T,Image1D<T> >(img);
    } else if (naxis == 2) {
        Image2D<T> img = read_image_data_2d<T>(f);
        return new Driver<T,Image2D<T> >(img);
    } else {
        std::cerr << "expected 1-dimensional or 2-dimensional data, got "
            << naxis << "-dimensional data" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


VDriver* from_image(const char* filename) {
    fitsfile* f = open_image_for_reading(filename);
    int s = 0;
    int bitpix = 0;
    fcheck(fits_get_img_type(f, &bitpix, &s));
    if (bitpix == get_fits_bitpix<float>()) {
        return from_image_helper<float>(f);
    } else if (bitpix == get_fits_bitpix<double>()) {
        return from_image_helper<double>(f);
    } else {
        std::cerr << "unexpected data type: ";
        if (bitpix < 0) {
            std::cerr << (-bitpix) << "-bit floats";
        } else {
            std::cerr << bitpix << "-bit integers";
        }
        std::cerr << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


template <typename T>
void write_image_helper(const char* filename, T* data, int size, int naxis, long *naxes)
{
    fitsfile* f = NULL;
    int s = 0;
    fcheck(fits_create_file(&f, filename, &s));
    fcheck(fits_create_img(f, get_fits_bitpix<T>(), naxis, naxes, &s));
    T nulval = std::numeric_limits<T>::quiet_NaN();
    fcheck(fits_write_imgnull(f, get_fits_type<T>(), 1, size, data, &nulval, &s));
    fcheck(fits_close_file(f, &s));
}


template <typename T>
void write_image(const char* filename, Image2D<T> img)
{
    long naxes[2] = {img.x, img.y};
    write_image_helper(filename, img.p, img.size(), 2, naxes);
}


template <typename T>
void write_image(const char* filename, Image1D<T> img)
{
    long naxes[1] = {img.x};
    write_image_helper(filename, img.p, img.size(), 1, naxes);
}


template void write_image(const char* filename, Image2D<float> img);
template void write_image(const char* filename, Image2D<double> img);
template void write_image(const char* filename, Image1D<float> img);
template void write_image(const char* filename, Image1D<double> img);
