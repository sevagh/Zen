#include <cuda/cuda.h>
#include <cuda/cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <thrust/complex.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/replace.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <hps/hps.h>
#include <hps/mfilt.h>
#include <libzen/hps.h>
#include <libzen/io.h>
#include <libzen/zen.h>

template <zen::Backend B>
zen::hps::HPRIOffline<B>::HPRIOffline(float fs,
                                     std::size_t hop_h,
                                     std::size_t hop_p,
                                     float beta_h,
                                     float beta_p,
                                     bool nocopybord)
    : io_h(zen::io::IOGPU(hop_h))
    , io_p(zen::io::IOGPU(hop_p))
    , hop_h(hop_h)
    , hop_p(hop_p)
{
	if (hop_h % hop_p != 0) {
		throw zen::ZgException("hop_h and hop_p should be evenly "
		                      "divisible");
	}

	p_impl_h = new zen::internal::hps::HPR<B>(
	    fs, hop_h, beta_h,
	    zen::internal::hps::HPSS_HARMONIC | zen::internal::hps::HPSS_PERCUSSIVE
	        | zen::internal::hps::HPSS_RESIDUAL,
	    zen::internal::hps::mfilt::MedianFilterDirection::TimeAnticausal,
	    !nocopybord);

	p_impl_p = new zen::internal::hps::HPR<B>(
	    fs, hop_p, beta_p, zen::internal::hps::HPSS_PERCUSSIVE,
	    zen::internal::hps::mfilt::MedianFilterDirection::TimeAnticausal,
	    !nocopybord);
}

template <zen::Backend B>
zen::hps::HPRIOffline<B>::HPRIOffline(float fs,
                                     std::size_t hop_h,
                                     std::size_t hop_p,
                                     float beta_h,
                                     float beta_p)
    : io_h(zen::io::IOGPU(hop_h))
    , io_p(zen::io::IOGPU(hop_p))
    , hop_h(hop_h)
    , hop_p(hop_p)
{
	if (hop_h % hop_p != 0) {
		throw zen::ZgException("hop_h and hop_p should be evenly "
		                      "divisible");
	}

	p_impl_h = new zen::internal::hps::HPR<B>(
	    fs, hop_h, beta_h,
	    zen::internal::hps::HPSS_HARMONIC | zen::internal::hps::HPSS_PERCUSSIVE
	        | zen::internal::hps::HPSS_RESIDUAL,
	    zen::internal::hps::mfilt::MedianFilterDirection::TimeAnticausal, true);

	p_impl_p = new zen::internal::hps::HPR<B>(
	    fs, hop_p, beta_p, zen::internal::hps::HPSS_PERCUSSIVE,
	    zen::internal::hps::mfilt::MedianFilterDirection::TimeAnticausal, true);
}

template <zen::Backend B>
zen::hps::HPRIOffline<B>::~HPRIOffline()
{
	delete p_impl_h;
	delete p_impl_p;
}

template <zen::Backend B>
zen::hps::HPRIOffline<B>::HPRIOffline(float fs,
                                     std::size_t hop_h,
                                     std::size_t hop_p)
    : HPRIOffline(fs, hop_h, hop_p, 2.0, 2.0){};

template <zen::Backend B>
zen::hps::HPRIOffline<B>::HPRIOffline(float fs)
    : HPRIOffline(fs, 4096, 256, 2.0, 2.0){};

static int
hpss_chunk_padder(std::vector<float>& audio, std::size_t hop, std::size_t lag)
{
	int n_chunks_ceil_iter
	    = ( int )(ceilf(( float )audio.size() / ( float )hop));
	int pad = n_chunks_ceil_iter * hop - audio.size();

	// pad with extra lag frames since offline/anticausal HPSS uses future
	// audio to produce past samples
	pad += lag * hop;

	// first lag frames are simply for prefilling the future frames of the stft
	n_chunks_ceil_iter += lag;

	audio.resize(audio.size() + pad, 0.0F);

	return n_chunks_ceil_iter;
}

template <>
std::array<std::vector<float>, 3>
zen::hps::HPRIOffline<zen::Backend::GPU>::process(std::vector<float> audio)
{
	// return same-sized vectors as a result
	int original_size = audio.size();

	int n_chunks_ceil_iter1
	    = hpss_chunk_padder(audio, p_impl_h->hop, p_impl_h->lag);

	// 2nd iteration uses xp1 + xr1 as the total content
	std::vector<float> intermediate(audio.size());
	std::vector<float> harmonic_out(audio.size());

	for (int i = 0; i < n_chunks_ceil_iter1; ++i) {
		// first apply HPR with large hop size for good harmonic separation
		// copy the song, hop_h samples at a time, into the large-sized
		// mapped/pinned IOGPU device
		thrust::copy(audio.begin() + i * hop_h,
		             audio.begin() + (i + 1) * hop_h, io_h.host_in);

		// process every large-sized hop
		p_impl_h->process_next_hop(io_h.device_in);

		// use xp1 + xr1 as input for the second iteration of HPR with small
		// hop size for good percussive separation
		thrust::transform(p_impl_h->percussive_out.begin(),
		                  p_impl_h->percussive_out.begin() + hop_h,
		                  p_impl_h->residual_out.begin(), io_h.device_out,
		                  zen::internal::hps::sum_vectors_functor());

		std::copy(io_h.host_out, io_h.host_out + hop_h,
		          intermediate.begin() + i * hop_h);

		thrust::copy(p_impl_h->harmonic_out.begin(),
		             p_impl_h->harmonic_out.begin() + hop_h, io_h.device_out);

		std::copy(io_h.host_out, io_h.host_out + hop_h,
		          harmonic_out.begin() + i * hop_h);
	}

	// offset intermediate by lag1*hop_h to skip the padded lag frames of
	// iter1
	std::copy(intermediate.begin() + p_impl_h->lag * hop_h, intermediate.end(),
	          intermediate.begin());
	std::copy(harmonic_out.begin() + p_impl_h->lag * hop_h, harmonic_out.end(),
	          harmonic_out.begin());

	intermediate.resize(original_size);
	harmonic_out.resize(original_size);
	audio.resize(original_size);

	int n_chunks_ceil_iter2
	    = hpss_chunk_padder(audio, p_impl_p->hop, p_impl_p->lag);
	std::vector<float> percussive_out(audio.size());
	std::vector<float> residual_out(audio.size());

	for (int i = 0; i < n_chunks_ceil_iter2; ++i) {
		// next apply HPR with small hop size for good harmonic separation
		// copy the song, hop_p samples at a time, into the small-sized
		// mapped/pinned IOGPU device
		std::copy(intermediate.begin() + i * hop_p,
		          intermediate.begin() + (i + 1) * hop_p, io_p.host_in);

		p_impl_p->process_next_hop(io_p.device_in);

		thrust::copy(p_impl_p->percussive_out.begin(),
		             p_impl_p->percussive_out.begin() + hop_p, io_p.device_out);

		std::copy(io_p.host_out, io_p.host_out + hop_p,
		          percussive_out.begin() + i * hop_p);

		thrust::copy(p_impl_p->residual_out.begin(),
		             p_impl_p->residual_out.begin() + hop_p, io_p.device_out);

		std::copy(io_p.host_out, io_p.host_out + hop_p,
		          residual_out.begin() + i * hop_p);
	}

	// rotate the useful part by lag2*hop_p to skip the padded lag frames of
	// iter2
	std::copy(percussive_out.begin() + p_impl_p->lag * hop_p,
	          percussive_out.end(), percussive_out.begin());

	std::copy(residual_out.begin() + p_impl_p->lag * hop_p, residual_out.end(),
	          residual_out.begin());

	// truncate all padding
	percussive_out.resize(original_size);
	residual_out.resize(original_size);

	return std::array<std::vector<float>, 3>{
	    harmonic_out, percussive_out, residual_out};
}

template <>
std::array<std::vector<float>, 3>
zen::hps::HPRIOffline<zen::Backend::CPU>::process(std::vector<float> audio)
{
	// return same-sized vectors as a result
	int original_size = audio.size();

	int n_chunks_ceil_iter1
	    = hpss_chunk_padder(audio, p_impl_h->hop, p_impl_h->lag);

	// 2nd iteration uses xp1 + xr1 as the total content
	std::vector<float> intermediate(audio.size());

	for (int i = 0; i < n_chunks_ceil_iter1; ++i) {
		// first apply HPR with large hop size for good harmonic separation
		p_impl_h->process_next_hop(audio.data() + i * hop_h);

		// use xp1 + xr1 as input for the second iteration of HPR with small
		// hop size for good percussive separation
		std::transform(p_impl_h->percussive_out.begin(),
		               p_impl_h->percussive_out.begin() + hop_h,
		               p_impl_h->residual_out.begin(),
		               intermediate.begin() + i * hop_h,
		               zen::internal::hps::sum_vectors_functor());
	}

	// offset intermediate by lag1*hop_h to skip the padded lag frames of
	// iter1
	std::copy(intermediate.begin() + p_impl_h->lag * hop_h, intermediate.end(),
	          intermediate.begin());

	intermediate.resize(original_size);
	audio.resize(original_size);

	int n_chunks_ceil_iter2
	    = hpss_chunk_padder(audio, p_impl_p->hop, p_impl_p->lag);
	std::vector<float> percussive_out(audio.size());

	for (int i = 0; i < n_chunks_ceil_iter2; ++i) {
		// next apply HPR with small hop size for good harmonic separation
		p_impl_p->process_next_hop(intermediate.data() + i * hop_p);

		std::copy(p_impl_p->percussive_out.begin(),
		          p_impl_p->percussive_out.begin() + hop_p,
		          percussive_out.begin() + i * hop_p);
	}

	// rotate the useful part by lag2*hop_p to skip the padded lag frames of
	// iter2
	std::copy(percussive_out.begin() + p_impl_p->lag * hop_p,
	          percussive_out.end(), percussive_out.begin());

	// truncate all padding
	percussive_out.resize(original_size);

	return std::array<std::vector<float>, 3>{
	    percussive_out, percussive_out, percussive_out};
}

template <zen::Backend B>
zen::hps::PRealtime<B>::PRealtime(float fs, std::size_t hop, float beta)
{
	p_impl = new zen::internal::hps::HPR<B>(
	    fs, hop, beta, zen::internal::hps::HPSS_PERCUSSIVE,
	    zen::internal::hps::mfilt::MedianFilterDirection::TimeCausal, true);
}

template <zen::Backend B>
zen::hps::PRealtime<B>::PRealtime(float fs,
                                 std::size_t hop,
                                 float beta,
                                 bool nocopybord)
{
	p_impl = new zen::internal::hps::HPR<B>(
	    fs, hop, beta, zen::internal::hps::HPSS_PERCUSSIVE,
	    zen::internal::hps::mfilt::MedianFilterDirection::TimeCausal,
	    !nocopybord);
}

template <zen::Backend B>
zen::hps::PRealtime<B>::~PRealtime()
{
	delete p_impl;
}

template <zen::Backend B>
zen::hps::PRealtime<B>::PRealtime(float fs, std::size_t hop)
    : PRealtime(fs, hop, 2.5){};

// (TODO: reassess) best-performing defaults
template <zen::Backend B>
zen::hps::PRealtime<B>::PRealtime(float fs)
    : PRealtime(fs, 256, 2.5){};

template <>
void zen::hps::PRealtime<zen::Backend::GPU>::process_next_hop(
    thrust::device_ptr<float> in_hop,
    thrust::device_ptr<float> out_hop)
{
	p_impl->process_next_hop(in_hop);
	thrust::copy(p_impl->percussive_out.begin(),
	             p_impl->percussive_out.begin() + p_impl->hop, out_hop);
}

template <>
void zen::hps::PRealtime<zen::Backend::CPU>::process_next_hop(float* in_hop,
                                                            float* out_hop)
{
	p_impl->process_next_hop(in_hop);
	std::copy(p_impl->percussive_out.begin(),
	          p_impl->percussive_out.begin() + p_impl->hop, out_hop);
}

template <>
void zen::hps::PRealtime<zen::Backend::GPU>::warmup(zen::io::IOGPU& io)
{
	int test_iters = 1000; // this is good enough in my experience
	std::vector<float> testdata(test_iters * p_impl->hop);
	std::vector<float> outdata(test_iters * p_impl->hop);
	std::iota(testdata.begin(), testdata.end(), 0.0F);

	for (std::size_t i = 0; i < test_iters; ++i) {
		thrust::copy(testdata.begin() + i * p_impl->hop,
		             testdata.begin() + (i + 1) * p_impl->hop, io.host_in);
		p_impl->process_next_hop(io.device_in);
		thrust::copy(io.host_out, io.host_out + p_impl->hop,
		             outdata.begin() + i * p_impl->hop);
	}

	p_impl->reset_buffers();
}

template <>
void zen::hps::PRealtime<zen::Backend::CPU>::warmup()
{
	int test_iters = 1000; // this is good enough in my experience
	std::vector<float> testdata(test_iters * p_impl->hop);
	std::vector<float> outdata(test_iters * p_impl->hop);
	std::iota(testdata.begin(), testdata.end(), 0.0F);

	for (std::size_t i = 0; i < test_iters; ++i) {
		p_impl->process_next_hop(testdata.data() + i * p_impl->hop);
		std::copy(p_impl->percussive_out.begin(),
		          p_impl->percussive_out.begin() + p_impl->hop,
		          outdata.data() + i * p_impl->hop);
	}

	p_impl->reset_buffers();
}

template <zen::Backend B>
void zen::internal::hps::HPR<B>::process_next_hop(InputPointer in_hop)
{
	// following the previous iteration
	// we rotate the percussive and harmonic arrays to get them ready
	// for the next hop and next overlap add
	if (output_percussive) {
		thrust::copy(percussive_out.begin() + hop, percussive_out.end(),
		             percussive_out.begin());
		thrust::fill(percussive_out.begin() + hop, percussive_out.end(), 0.0);
	}
	if (output_harmonic) {
		thrust::copy(harmonic_out.begin() + hop, harmonic_out.end(),
		             harmonic_out.begin());
		thrust::fill(harmonic_out.begin() + hop, harmonic_out.end(), 0.0);
	}
	if (output_residual) {
		thrust::copy(residual_out.begin() + hop, residual_out.end(),
		             residual_out.begin());
		thrust::fill(residual_out.begin() + hop, residual_out.end(), 0.0);
	}

	// append latest hop samples e.g. input = input[hop:] + current_hop
	thrust::copy(input.begin() + hop, input.end(), input.begin());
	thrust::copy(in_hop, in_hop + hop, input.begin() + hop);

	// populate curr_fft with input .* square root von hann window
	thrust::transform(input.begin(), input.end(), window.window.begin(),
	                  fft.fft_vec.begin(), zen::internal::hps::window_functor());

	// zero out the second half of the fft
	thrust::fill(fft.fft_vec.begin() + nwin, fft.fft_vec.end(),
	             thrust::complex<float>{0.0, 0.0});

	// perform the fft
	fft.forward();

	// rotate stft matrix to move the oldest column to the end
	// copy curr_fft into the last column of the stft
	thrust::copy(
	    sliding_stft.begin() + nfft, sliding_stft.end(), sliding_stft.begin());
	thrust::copy(
	    fft.fft_vec.begin(), fft.fft_vec.end(), sliding_stft.end() - nfft);

	// calculate the magnitude of the stft
	thrust::transform(sliding_stft.begin(), sliding_stft.end(), s_mag.begin(),
	                  zen::internal::hps::complex_abs_functor());

	// up to this stage, median filter + SSE filter are identical
	switch (filter_type) {
		case FilterType::Median:
			// apply median filter in horizontal and vertical directions with NPP
			// to create percussive and harmonic spectra
			apply_median_filter();
			break;
		case FilterType::SSE:
			// apply a Stochastic Spectrum Estimation filter
			// to create transient and steady state spectra
			apply_sse_filter();
			break;
	};

	if (output_percussive) {
		// compute percussive mask from harmonic + percussive magnitude spectra
		thrust::transform(percussive_matrix.end() - lag * nfft,
		                  percussive_matrix.end() - (lag - 1) * nfft,
		                  harmonic_matrix.end() - lag * nfft,
		                  percussive_mask.end() - lag * nfft,
		                  zen::internal::hps::mask_functor(beta));

		// apply lag column of percussive mask to recover percussive audio from
		// original fft
		thrust::transform(sliding_stft.end() - lag * nfft,
		                  sliding_stft.end() - (lag - 1) * nfft,
		                  percussive_mask.end() - lag * nfft,
		                  fft.fft_vec.begin(),
		                  zen::internal::hps::apply_mask_functor());
		fft.backward();

		// now curr_fft has the current iteration's fresh samples
		// we overlap-add it the real part to the previous
		thrust::transform(fft.fft_vec.begin(), fft.fft_vec.begin() + nwin,
		                  percussive_out.begin(), percussive_out.begin(),
		                  zen::internal::hps::overlap_add_functor(COLA_factor));
	}

	if (output_harmonic) {
		// compute harmonic mask from harmonic + percussive magnitude spectra
		thrust::transform(harmonic_matrix.end() - lag * nfft,
		                  harmonic_matrix.end() - (lag - 1) * nfft,
		                  percussive_matrix.end() - lag * nfft,
		                  harmonic_mask.end() - lag * nfft,
		                  zen::internal::hps::mask_functor(beta - Eps));

		thrust::transform(sliding_stft.end() - lag * nfft,
		                  sliding_stft.end() - (lag - 1) * nfft,
		                  harmonic_mask.end() - lag * nfft, fft.fft_vec.begin(),
		                  zen::internal::hps::apply_mask_functor());

		fft.backward();

		thrust::transform(fft.fft_vec.begin(), fft.fft_vec.begin() + nwin,
		                  harmonic_out.begin(), harmonic_out.begin(),
		                  zen::internal::hps::overlap_add_functor(COLA_factor));
	}

	if (output_residual) {
		// compute residual mask from harmonic and percussive masks
		thrust::transform(harmonic_mask.begin(), harmonic_mask.end(),
		                  percussive_mask.begin(), residual_mask.begin(),
		                  zen::internal::hps::residual_mask_functor());

		thrust::transform(sliding_stft.end() - lag * nfft,
		                  sliding_stft.end() - (lag - 1) * nfft,
		                  residual_mask.end() - lag * nfft, fft.fft_vec.begin(),
		                  zen::internal::hps::apply_mask_functor());

		fft.backward();

		thrust::transform(fft.fft_vec.begin(), fft.fft_vec.begin() + nwin,
		                  residual_out.begin(), residual_out.begin(),
		                  zen::internal::hps::overlap_add_functor(COLA_factor));
	}
}

template <zen::Backend B>
void zen::internal::hps::HPR<B>::apply_median_filter() {
	time.filter(s_mag, harmonic_matrix);
	frequency.filter(s_mag, percussive_matrix);
}

template <zen::Backend B>
void zen::internal::hps::HPR<B>::apply_sse_filter() {
	// calculate the reciprocal of the magnitude stft
	// SSE paper calls it a power spectrogram but there's no ^2 factor
	thrust::transform(s_mag.begin(), s_mag.end(), reciprocal.begin(),
	                  zen::internal::hps::reciprocal_functor());

	// now do SSE in the time and frequency directions
	// causality affects whether we can go forward in the time direction
	// same as median filtering
	time_sse.filter(reciprocal, harmonic_matrix);
	frequency_sse.filter(reciprocal, percussive_matrix);

	// take reciprocal again for the final percussive/harmonic magnitude spectra
	thrust::transform(percussive_matrix.begin(), percussive_matrix.end(), reciprocal.begin(),
	                  zen::internal::hps::reciprocal_functor());
	thrust::transform(harmonic_matrix.begin(), harmonic_matrix.end(), reciprocal.begin(),
	                  zen::internal::hps::reciprocal_functor());

}

template class zen::hps::HPRIOffline<zen::Backend::CPU>;
template class zen::hps::HPRIOffline<zen::Backend::GPU>;

template class zen::hps::PRealtime<zen::Backend::CPU>;
template class zen::hps::PRealtime<zen::Backend::GPU>;
