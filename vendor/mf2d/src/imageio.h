#ifndef IMAGEIO_H
#define IMAGEIO_H

#include "driver.h"

template <typename T>
void write_image(const char* filename, Image2D<T> img);

template <typename T>
void write_image(const char* filename, Image1D<T> img);

VDriver* from_image(const char* filename);

#endif
