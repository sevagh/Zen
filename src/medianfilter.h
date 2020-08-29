#ifndef MEDIAN_FILTER_PRIVATE_H
#define MEDIAN_FILTER_PRIVATE_H

#include <complex>
#include <cstddef>
#include <vector>

#include <cufft.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "npp.h"
#include "nppdefs.h"
#include "nppi.h"

#include "rhythm_toolkit/rhythm_toolkit.h"

// TODO rename freq/time to x/y vertical/horizontal

namespace rhythm_toolkit_private {
namespace median_filter {

	enum MedianFilterDirection {
		Time,
		Frequency,
	};

	class MedianFilterGPU {
	public:
		int time;
		int frequency;
		int filter_len;

		NppiSize roi;
		int filter_mid;
		NppiSize mask;
		NppiPoint anchor;
		Npp8u* buffer;
		Npp32u buffer_size;

		int nstep;
		int start_pixel_offset;

		// use time and frequency as axis names
		MedianFilterGPU(int time, int frequency, int filter_len, MedianFilterDirection dir)
		: time(time)
		, frequency(frequency)
		, filter_len(filter_len)
		, nstep(frequency * sizeof(Npp32f)) // expect 1D linear memory layout e.g. i*y + j
		{
			if ((dir == MedianFilterDirection::Time && filter_len > time) ||  (dir == MedianFilterDirection::Frequency && filter_len > frequency)) {
				throw rhythm_toolkit::RtkException("median filter bigger than matrix dimension");
			}

			filter_len += (1 - (filter_len % 2)); // make sure filter length is odd
			filter_mid = ( int )floorf(( float )filter_len / 2);


			switch (dir) {
				// https://docs.nvidia.com/cuda/npp/nppi_conventions_lb.html#roi_specification
				case MedianFilterDirection::Time:
					roi = NppiSize{frequency, time-filter_len};

					mask = NppiSize{1, filter_len};
					anchor = NppiPoint{0, filter_mid};

					start_pixel_offset = filter_mid * nstep;

					break;
				case MedianFilterDirection::Frequency:
					roi = NppiSize{frequency-filter_len, time};

					mask = NppiSize{filter_len, 1};
					anchor = NppiPoint{filter_mid, 0};

					start_pixel_offset = filter_mid;
					break;
			}

			NppStatus npp_status = nppiFilterMedianGetBufferSize_32f_C1R(
			    roi, mask, &buffer_size);
			if (npp_status != NPP_NO_ERROR) {
				std::cerr << "NPP error " << npp_status << std::endl;
				std::exit(1);
			}

			cudaError cuda_status
			    = cudaMalloc(( void** )&buffer, buffer_size);
			if (cuda_status != cudaSuccess) {
				std::cerr << cudaGetErrorString(cuda_status);
				std::exit(1);
			}
		};

		~MedianFilterGPU()
		{
			cudaFree(buffer);
		}

		void filter(Npp32f *src, Npp32f *dst) {
			nppiFilterMedian_32f_C1R(src + start_pixel_offset, nstep,
				 dst + start_pixel_offset,
				 nstep, roi, mask, anchor, buffer);

		}
	};
}; // namespace median_filter
}; // namespace rhythm_toolkit_private

#endif /* MEDIAN_FILTER_PRIVATE_H */
