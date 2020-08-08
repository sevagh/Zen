#include <cstdlib>
#include <iostream>
#include "driver.h"
#include "imageio.h"

int main(int argc, const char** argv) {
    if (argc != 5) {
        std::cerr << "usage: " << argv[0] << " radius input output-median output-difference" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    int h = atoi(argv[1]);
    if (h < 0) {
        std::cerr << "radius has to be at least 0" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    const char* source = argv[2];
    const char* target_med = argv[3];
    const char* target_diff = argv[4];

    VDriver* driver = from_image(source);
    driver->process(h);
    driver->write(target_med);
    driver->diff();
    driver->write(target_diff);
    delete driver;
}
