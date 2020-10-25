#ifndef BTRACK_CIRCULARBUFFER_H
#define BTRACK_CIRCULARBUFFER_H

#include <cstddef>
#include <array>

static constexpr bool is_power_of_two(std::size_t value) {
    return value != 0 && !(value & (value - 1));
}

template <std::size_t N>
class CircularBuffer {
static_assert(is_power_of_two(N));

public:
std::array<float, N> buffer = {};

CircularBuffer() : writeIndex(0) {};

float &operator[](std::size_t i)
{
    return buffer[(i + writeIndex) & (N-1)];
}

void append(float v)
{
    buffer[writeIndex] = v;
    writeIndex = (writeIndex + 1) & (N-1);
}

private:
std::size_t writeIndex;
};

#endif
