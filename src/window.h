#ifndef WINDOW_PRIVATE_H
#define WINDOW_PRIVATE_H

#include <array>
#include <cstddef>
#include <math.h>
#include <thrust/device_vector.h>
#include <vector>

#include "rhythm_toolkit/rhythm_toolkit.h"

namespace rhythm_toolkit_private {
namespace window {

	static constexpr float PI = 3.14159265359F;

	enum WindowType {
		SqrtVonHann,
	};

	template <typename T>
	class Window {
	public:
		T window;

		Window(WindowType window_type, std::size_t window_size)
		    : window(window_size, 0.0)
		{
			switch (window_type) {
			default: // only implement a von Hann window for now

				// typically this would be "window_size-1"
				// but we want the behavior of a matlab 'periodic' hann
				// vs. the default 'symm' hann
				auto N = ( float )(window_size);

				for (std::size_t n = 0; n < window_size; ++n) {
					window[n] = sqrtf(
					    0.5F * (1.0F - cosf(2.0F * PI * ( float )n / N)));
				}
			}
		}
	};
	using WindowGPU = Window<thrust::device_vector<float>>;
	using WindowCPU = Window<std::vector<float>>;
}; // namespace window
}; // namespace rhythm_toolkit_private

#endif // WINDOW_PRIVATE_H
