2D Median Filter for Floating-Point Data
========================================

Alpha version, bugs are possible, use at your own risk.



Usage
-----

### As a command-line tool

You can use "mf2d" to process 32-bit and 64-bit FITS images,
both 1-dimensional and 2-dimensional.

Usage:

    bin/mf2d radius input output-median output-difference

Example:

    bin/mf2d 15 example/test-1.fits a.fits b.fits

"Radius" is the window radius in pixels. For example, a radius
of 15 in a 2D image means a window of 31x31 pixels in total.

"Input" is a FITS file, 1 or 2 dimensions, 32-bit or 64-bit
floating point values.

"Output-median" will be a FITS file that contains the result
of median filtering.

"Output-difference" will be another FITS file that contains
the difference between "input" and "output-median".

All file names are given in a format supported by the cfitsio
library. In particular, prefix the output file names with an
exclamation mark if you want to overwrite existing files:

    bin/mf2d 15 example/test-1.fits '!a.fits' '!b.fits'

You can also filter just a small part of a large image:

    bin/mf2d 15 'example/test-1.fits[1:100,1:200]' a.fits b.fits


### As a subroutine

You can use this software as a subroutine to filter 1-dimensional
and 2-dimensional arrays of floating point values.

You will only need filter.cc and filter.h; you do not need
any additional libraries.

See filter.h for the interface, and mf2d-unittest.cc for
simple examples.



Performance
-----------

Some examples of typical running times on my Macbook Air laptop
(1.7 GHz Intel Haswell, 2 cores, 4 threads):

  - less than 0.2s in total:

        bin/mf2d 10 example/test-1.fits '!a.fits' '!b.fits'

  - less than 0.4s in total:

        bin/mf2d 40 example/test-1.fits '!a.fits' '!b.fits'

  - less than 0.6s in total:

        bin/mf2d 80 example/test-1.fits '!a.fits' '!b.fits'

  - less than 0.8s in total:

        bin/mf2d 120 example/test-1.fits '!a.fits' '!b.fits'

The sample file is 1024x1024 pixels, 32-bit floats.



Details
-------

### Missing values

Missing values (NaNs) are treated as missing values. For
example, the median of [x, y, NaN, z] is the same as the
median of [x, y, z]. In the output, there is a NaN if and
only if the entire window is empty (only NaNs).


### Boundaries

Boundaries are handled by clipping the sliding window to
image boundaries. For example, while middle parts of the output
will be medians of (2r+1) x (2r+1) boxes, the corners of the
output will be medians of (r+1) x (r+1) boxes.

In the middle parts, the window always contains an odd number
of pixels, and hence the median is unique. Near the boundaries
we may have an even number of pixels in the window; in those
cases we will output the average of the two middle values.


### Limits

The total image size has to be less than 2^31 pixels.



Compiling
---------

Install cfitsio first (see below for details).
Then compile mf2d as follows, depending on your platform.


### Linux and GCC

Compile:

    compile/gcc-linux.sh

Test:

    test/test.sh


### Linux and ICC

Compile:

    compile/gcc-linux.sh

Test:

    test/test.sh


### OS X and GCC

You can get GCC 4.9 from Homebrew:

    brew install gcc

Set CXX to point to the right compiler:

    export CXX=g++-4.9

Compile:

    compile/gcc-osx.sh

Test:

    test/test.sh


### OS X and Clang

(Not recommended: slow, will not use OpenMP.)

Compile:

    compile/clang-osx.sh

Test:

    test/test.sh


### Meson and Ninja

See util/build-setup and util/build-all.



Installing cfitsio
------------------

To compile the command-line tool, you will need cfitsio.
On OS X, you can use Homebrew:

    brew install cfitsio

On Ubuntu Linux, you can try:

    apt-get install libcfitsio3-dev

Alternatively, you can download the source from
http://heasarc.gsfc.nasa.gov/fitsio/fitsio.html
and compile and install it e.g. as follows:

    ./configure --prefix=$HOME/opt
    make
    make install

If you use a nonstandard location, set the paths accordingly
so that the compiler and linker can find it:

    export CPATH=$HOME/opt/include
    export LIBRARY_PATH=$HOME/opt/lib



Platforms and versions
----------------------

Tested on the following platforms:

  - OS X 10.10
  - Ubuntu 12.04
  - Ubuntu 14.04

With e.g. the following compilers:

  - GCC 4.4, 4.6, 4.7, 4.8, 4.9
  - ICC 13.0, 14.0, 15.0
  - Apple LLVM version 6.0

Using the following libraries:

  - cfitsio 3.370



License
-------

Copyright (c) 2014, Jukka Suomela.

You can distribute and use this software under the MIT license:
http://opensource.org/licenses/MIT

To contact the author, see https://jukkasuomela.fi/



Acknowledgements
----------------

Test data contributed by Jean-Eric Campagne.
