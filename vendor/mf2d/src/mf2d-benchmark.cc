#include <cstdlib>
#include <iostream>
#include "driver.h"
#include "imageio.h"

int main(int argc, const char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " input" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    const char* source = argv[1];
    VDriver* driver = from_image(source);
    driver->benchmark();
    delete driver;
}
