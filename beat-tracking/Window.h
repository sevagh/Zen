#ifndef BTRACK_WINDOW_H
#define BTRACK_WINDOW_H

#include "gcem.hpp"
#include <array>
#include <cstddef>

static constexpr float PI = 3.14159265359F;

template <std::size_t WindowSize>
struct Window {
	float data[WindowSize];
};

namespace detail {
	template <std::size_t WindowSize>
	constexpr Window<WindowSize> calculate_hamming_window()
	{
		auto N = ( float )(WindowSize - 1);
		float n_val = 0.0F;

		Window<WindowSize> window = {};
		for (size_t n = 0; n < WindowSize; ++n) {
			window.data[n]
			    = 0.54F - (0.46F * gcem::cos(2.0F * PI * ((n_val) / N)));
			n_val += 1.0F;
		}
		return window;
	}

	template <std::size_t WindowSize>
	constexpr Window<WindowSize> calculate_hanning_window()
	{
		auto N = ( float )(WindowSize - 1);

		Window<WindowSize> window = {};
		for (size_t n = 0; n < WindowSize; ++n) {
			window.data[n] = 0.5F * (1.0F - gcem::cos(2.0F * PI * (n / N)));
		}
		return window;
	}
	template <std::size_t WindowSize>
	constexpr Window<WindowSize> calculate_blackman_window()
	{
		auto N = ( float )(WindowSize - 1);
		float n_val = 0.0F;

		Window<WindowSize> window = {};
		for (size_t n = 0; n < WindowSize; ++n) {
			{
				window.data[n]
				    = 0.42F - (0.5F * gcem::cos(2.0F * PI * (n_val / N)))
				      + (0.08F * gcem::cos(4.0F * PI * (n_val / N)));
				n_val = n_val + 1.0F;
			}
		}

		return window;
	}

	template <std::size_t WindowSize>
	constexpr Window<WindowSize> calculate_tukey_window()
	{
		auto N = ( float )(WindowSize - 1);
		float alpha = 0.5F;
		int width = ( int )(gcem::floor(alpha * (N) / 2.0F));
		Window<WindowSize> window = {};

		for (int n = 0; n <= width; ++n) {
			window.data[n]
			    = 0.5F
			      * (1.0F
				 + gcem::cos(PI
					     * (-1.0F + 2.0F * (( float )n) / alpha / N)));
		}
		for (int n = width + 1; n < WindowSize - width - 1; ++n) {
			window.data[n] = 1.0F;
		}
		for (int n = WindowSize - width - 1; n < WindowSize; ++n) {
			window.data[n]
			    = 0.5F
			      * (1.0F
				 + gcem::cos(PI
					     * (-2.0F / alpha + 1.0F
						+ 2.0F * (( float )n) / alpha / N)));
		}

		return window;
	}

	template <std::size_t WindowSize>
	constexpr Window<WindowSize> calculate_rectangular_window()
	{
		Window<WindowSize> window = {};

		for (size_t n = 0; n < WindowSize; ++n) {
			window.data[n] = 1.0;
		}
		return window;
	}
} // namespace detail

enum WindowType {
	RectangularWindow,
	HanningWindow,
	HammingWindow,
	BlackmanWindow,
	TukeyWindow
};

template <std::size_t WindowSize>
constexpr Window<WindowSize> Hamming = detail::calculate_hamming_window<WindowSize>();

template <std::size_t WindowSize>
constexpr Window<WindowSize> Hanning = detail::calculate_hanning_window<WindowSize>();

template <std::size_t WindowSize>
constexpr Window<WindowSize> Blackman = detail::calculate_blackman_window<WindowSize>();

template <std::size_t WindowSize>
constexpr Window<WindowSize> Tukey = detail::calculate_tukey_window<WindowSize>();

template <std::size_t WindowSize>
constexpr Window<WindowSize> Rectangular
    = detail::calculate_rectangular_window<WindowSize>();

template <std::size_t WindowSize, WindowType windowType>
constexpr Window<WindowSize> get_window()
{
	switch (windowType) {
	case HammingWindow:
		return Hamming<WindowSize>;
	case HanningWindow:
		return Hanning<WindowSize>;
	case BlackmanWindow:
		return Blackman<WindowSize>;
	case TukeyWindow:
		return Tukey<WindowSize>;
	case RectangularWindow:
		return Rectangular<WindowSize>;
	default:
		return Hanning<WindowSize>;
	}
}

#endif
