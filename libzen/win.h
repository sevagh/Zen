#ifndef ZG_WIN_INTERNAL_H
#define ZG_WIN_INTERNAL_H

#include <array>
#include <cstddef>
#include <math.h>
#include <thrust/device_vector.h>
#include <vector>

namespace zen {
namespace internal {
	namespace win {

		static constexpr float PI = 3.14159265359F;

		enum WindowType {
			SqrtVonHann,
			VonHann,
		};

		template <typename T>
		class Window {
		public:
			T window;

			Window(WindowType window_type, std::size_t window_size)
			    : window(window_size, 0.0)
			{
				switch (window_type) {
				case WindowType::SqrtVonHann: {
					// typically this would be "window_size-1"
					// but we want the behavior of a matlab 'periodic' hann
					// vs. the default 'symm' hann
					auto N = ( float )(window_size);

					for (std::size_t n = 0; n < window_size; ++n) {
						window[n] = sqrtf(
						    0.5F * (1.0F - cosf(2.0F * PI * ( float )n / N)));
					}
				} break;
				case WindowType::VonHann: {
					auto N = ( float )(window_size);

					for (std::size_t n = 0; n < window_size; ++n) {
						window[n]
						    = 0.5F * (1.0F - cosf(2.0F * PI * ( float )n / N));
					}
				} break;
				}
			}
		};
		using WindowGPU = Window<thrust::device_vector<float>>;
		using WindowCPU = Window<std::vector<float>>;
	}; // namespace win
};     // namespace internal
};     // namespace zen

#endif // ZG_WIN_INTERNAL_H
