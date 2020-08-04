#ifndef WINDOW_H
#define WINDOW_H

#include <array>
#include <cstddef>
#include <vector>
#include <math.h>

namespace rhythm_toolkit {
namespace window {

	static constexpr float PI = 3.14159265359F;

	enum WindowType {
		VonHann,
	};

	class Window {
	public:
		std::vector<float> window;

		Window(WindowType window_type, std::size_t window_size)
		    : window(std::vector<float>(window_size, 0.0))
		{
			switch (window_type) {
			default: // only implement a von Hann window for now
				auto N = ( float )(window_size - 1);

				for (std::size_t n = 0; n < window_size; ++n) {
					window[n] = 0.5F * (1.0F - cosf(2.0F * PI * (n / N)));
				}
			}
		}
	};

}; // namespace window
}; // namespace rhythm_toolkit

#endif // WINDOW_H
