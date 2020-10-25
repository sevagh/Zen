#include "pitch_detection.h"
#include <algorithm>
#include <complex>
#include <float.h>
#include <ipp.h>
#include <ippdefs.h>
#include <ippi.h>
#include <map>
#include <numeric>
#include <vector>

#define MPM_CUTOFF 0.93
#define MPM_SMALL_CUTOFF 0.5
#define MPM_LOWER_PICH_CUTOFF 80.0

std::pair<float, float> parabolic_interpolation(const std::vector<float>& array,
                                                int x_)
{
	int x_adjusted;
	float x = ( float )x_;

	if (x < 1) {
		x_adjusted = (array[x] <= array[x + 1]) ? x : x + 1;
	}
	else if (x > signed(array.size()) - 1) {
		x_adjusted = (array[x] <= array[x - 1]) ? x : x - 1;
	}
	else {
		float den = array[x + 1] + array[x - 1] - 2 * array[x];
		float delta = array[x - 1] - array[x + 1];
		return (!den) ? std::make_pair(x, array[x])
		              : std::make_pair(x + delta / (2 * den),
		                               array[x] - delta * delta / (8 * den));
	}
	return std::make_pair(x_adjusted, array[x_adjusted]);
}

void real_autocorrelation(const std::vector<float>& audio_buffer, MPM* mpm)
{
	if (audio_buffer.size() == 0)
		throw std::invalid_argument("audio_buffer shouldn't be empty");

	std::transform(audio_buffer.begin(), audio_buffer.begin() + mpm->N,
	               mpm->out_im.begin(), [](float x) -> std::complex<float> {
		               return std::complex<float>(x, static_cast<float>(0.0));
	               });

	ippsFFTFwd_CToC_32fc_I(
	    ( Ipp32fc* )mpm->out_im.data(), mpm->fft_spec, mpm->p_mem_buffer);

	std::complex<float> scale
	    = {1.0f / ( float )(mpm->N * 2), static_cast<float>(0.0)};
	for (int i = 0; i < mpm->N; ++i)
		mpm->out_im[i] *= std::conj(mpm->out_im[i]) * scale;

	ippsFFTInv_CToC_32fc_I(
	    ( Ipp32fc* )mpm->out_im.data(), mpm->fft_spec, mpm->p_mem_buffer);

	std::transform(
	    mpm->out_im.begin(), mpm->out_im.begin() + mpm->N,
	    mpm->out_real.begin(),
	    [](std::complex<float> cplx) -> float { return std::real(cplx); });
}

static std::vector<int> peak_picking(const std::vector<float>& nsdf)
{
	std::vector<int> max_positions{};
	int pos = 0;
	int cur_max_pos = 0;
	ssize_t size = nsdf.size();

	while (pos < (size - 1) / 3 && nsdf[pos] > 0)
		pos++;
	while (pos < size - 1 && nsdf[pos] <= 0.0)
		pos++;

	if (pos == 0)
		pos = 1;

	while (pos < size - 1) {
		if (nsdf[pos] > nsdf[pos - 1] && nsdf[pos] >= nsdf[pos + 1]
		    && (cur_max_pos == 0 || nsdf[pos] > nsdf[cur_max_pos])) {
			cur_max_pos = pos;
		}
		pos++;
		if (pos < size - 1 && nsdf[pos] <= 0) {
			if (cur_max_pos > 0) {
				max_positions.push_back(cur_max_pos);
				cur_max_pos = 0;
			}
			while (pos < size - 1 && nsdf[pos] <= 0.0) {
				pos++;
			}
		}
	}
	if (cur_max_pos > 0) {
		max_positions.push_back(cur_max_pos);
	}
	return max_positions;
}

float MPM::pitch(const std::vector<float>& audio_buffer)
{
	real_autocorrelation(audio_buffer, this);

	std::vector<int> max_positions = peak_picking(this->out_real);
	std::vector<std::pair<float, float>> estimates;

	float highest_amplitude = -DBL_MAX;

	for (int i : max_positions) {
		highest_amplitude = std::max(highest_amplitude, this->out_real[i]);
		if (this->out_real[i] > MPM_SMALL_CUTOFF) {
			auto x = parabolic_interpolation(this->out_real, i);
			estimates.push_back(x);
			highest_amplitude = std::max(highest_amplitude, std::get<1>(x));
		}
	}

	if (estimates.empty())
		return -1;

	float actual_cutoff = MPM_CUTOFF * highest_amplitude;
	float period = 0;

	for (auto i : estimates) {
		if (std::get<1>(i) >= actual_cutoff) {
			period = std::get<0>(i);
			break;
		}
	}

	float pitch_estimate = (sample_rate / period);

	this->clear();

	return (pitch_estimate > MPM_LOWER_PICH_CUTOFF) ? pitch_estimate : -1;
}
