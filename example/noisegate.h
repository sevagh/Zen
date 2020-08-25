#include <cmath>
#include <deque>
#include <boost/circular_buffer.hpp>
#include <cstddef>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>

using namespace std;

// A sliding window that maintains its maximum absolute value
class MaxWindow {
  private:
    // Window size.
    deque<float>::size_type window_size;
    // Samples within the window.
    boost::circular_buffer<float> buf;
    // Indinces into the whole track (not buf!), corresponding to the decreasing
    // subsequence of samples within the current window.
    deque<unsigned long> indices;
    // Total cumulative number of samples pushed into this window.
    // Used to convert 'indices' to actual buf indices.
    unsigned long n_samples = 0;
    // Get sample value by its absolute index.
    inline float get_sample(unsigned long index) const {
      return buf[buf.size() - (n_samples - index)];
    }
  public:
    MaxWindow(deque<float>::size_type window_size)
      : window_size(window_size), buf(window_size) {};
    void push(float sample) {
      sample = abs(sample);
      while (!indices.empty() && get_sample(indices.back()) <= sample) {
        indices.pop_back();
      }
      while (!indices.empty() && indices.front() <= n_samples - window_size) {
        indices.pop_front();
      }
      indices.push_back(n_samples++);
      buf.push_back(sample);
    }
    float level() const {
      return get_sample(indices.front());
    }
};

// A sliding window that knows at each moment how much non-silence it contains.
//
// The window has the latency ns_window_size.
class NonSilenceWindow {
  private:
    boost::circular_buffer<bool> buf; // true == non-silent
    MaxWindow max_window;
    float sample_rate;
    float level_threshold; // a threshold above which the sound is considered non-silent
    unsigned long nonsilent_samples = 0;
  public:
    NonSilenceWindow(boost::circular_buffer<float>::capacity_type ns_window_size,
                     boost::circular_buffer<float>::capacity_type max_window_size,
                     float sample_rate,
                     float level_threshold)
      : buf(ns_window_size), max_window(max_window_size),
        sample_rate(sample_rate), level_threshold(level_threshold)
      {};
    void push(float sample) {
      max_window.push(sample);
      if (buf.full())
        nonsilent_samples -= buf.front();
      bool new_nonsilent = max_window.level() >= level_threshold;
      buf.push_back(new_nonsilent);
      nonsilent_samples += new_nonsilent;
    }
    // Get the total amount of non-silence inside the window in seconds
    float nonsilent() const {
      return nonsilent_samples / sample_rate;
    }
};

// A window that smoothes the transition between the open and closed states of
// the gate.
//
// The state of the gate is represented by a bool: true = open, false = closed.
//
// When the gate moves from open to closed (true -> false), the gate closes
// smoothly after that.
//
// When the gate moves from closed to open (false -> true), this event is
// anticipated ahead of time and the transition is again smoothed.
//
// This function could probably be optimized by introducing more states and
// avoiding multiplications when the gate remains open or closed for a long time.
class SmoothingWindow {
  private:
    const float floor = 1e-4; // -80 dB
    unsigned long window_size;
    // The current scaling factor applied to the sound samples.
    float current_coef = 1;
    // Are we currently rising (true) or falling (false)?
    bool rising = true;
    // The number of samples since we've last seen the gate open.
    // If it's more than the window size, we may begin to decrease the scaling
    // factor.
    long unsigned samples_since_open = 0;
    // factor is initialized in the constructor based on
    // the window size and then never changes.
    float factor;
  public:
    SmoothingWindow(unsigned long window_size)
      : window_size(window_size),
        factor(exp(-log(floor)/window_size))
      {}
    // Push a new sample (is the gate open?)
    void push(bool open) {
      if (open) {
        samples_since_open = 0;
        rising = true;
      } else {
        samples_since_open++;
        if (samples_since_open > window_size) {
          rising = false;
        }
      }
      if (rising) {
        current_coef = min(max(current_coef, floor) * factor, 1.f);
      } else {
        current_coef = current_coef / factor;
        if (current_coef < floor) {
          current_coef = 0.f;
        }
      }
    }
    // Get the current scaling factor (with the latency equal to the
    // attack/decay duration)
    float scaling_factor() const {
      return current_coef;
    }
};

const unsigned long port_count = 7;

class NoiseGate {
public:
  unsigned sample_rate;
  unique_ptr<NonSilenceWindow> ns_window;
  unique_ptr<SmoothingWindow>  sm_window;
  unique_ptr<boost::circular_buffer<float>> buf;

  // NB: we cannot do much initialization in the constructor because the ports
  // may be connected after it is called.
  NoiseGate(unsigned sample_rate)
    : sample_rate(sample_rate) {}

  void run(unsigned long n_samples, float *input, float *output) {

	  float threshold_dB = -0.0f;
	  int window_size_ms = 3; // match hpss
	  int nonsilent_size_ms = 3;
	  int attack_ms = 0.01;

    float threshold     = pow(10.f, threshold_dB) / 20.f;
    float window_size   = (float)window_size_ms / 1000.0f; // in seconds
    float min_nonsilent = (float)nonsilent_size_ms / 1000.0f; // in seconds
    float attack        = (float)attack_ms / 1000.0f; // in seconds

    unsigned half_window_samples = window_size * sample_rate / 2.f;
    unsigned window_samples = 2 * half_window_samples + 1;
    unsigned sm_window_size = attack * sample_rate;
    unsigned latency_samples = half_window_samples + sm_window_size;

    if (ns_window == nullptr) {
      ns_window = make_unique<NonSilenceWindow>(window_samples,
                                                 sample_rate * 5e-3,
                                                 sample_rate,
                                                 threshold);
    }
    if (sm_window == nullptr) {
      sm_window = make_unique<SmoothingWindow>(sm_window_size);
    }
    if (buf == nullptr) {
      buf = make_unique<boost::circular_buffer<float>>(latency_samples);
    }
    for (unsigned i = 0; i < n_samples; i++) {
      // save the sample so we don't lose it after writing to output[i]
      float sample = input[i];
      ns_window->push(sample);
      sm_window->push(ns_window->nonsilent() >= min_nonsilent);
      if (buf->full()) {
        output[i] = buf->front() * sm_window->scaling_factor();
      }
      else {
        output[i] = 0;
      }
      buf->push_back(sample);
    }
  }
};
