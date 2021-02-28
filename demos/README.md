## Performance

### Pre-processing for pitch tracking

The [pitch-tracking](./pitch-tracking) demo applies real-time harmonic separation with a hop size of 4096 (optimal for harmonic results), and applies the McLeod Pitch Method on a stream of 4096-sized chunks from a wav file.

The clip [samples/simple_mixed.wav](./samples/simple_mixed.wav) contains a viola playing an E3 with a superimposed drum track. The demo results show that the pitch tracking is improved with real-time harmonic separation. **Verdict:** :heavy_check_mark:
```
$ ./build/pitch-tracking/pitch-track ./samples/simple_mixed.wav
t: 1.11,        pitch (+HPR): 163.11,   pitch (-HPR): 741.54
t: 1.21,        pitch (+HPR): 163.18,   pitch (-HPR): 2164.02
t: 1.30,        pitch (+HPR): 163.26,   pitch (-HPR): 177.98
t: 1.39,        pitch (+HPR): 163.29,   pitch (-HPR): 217.32
t: 1.49,        pitch (+HPR): 163.30,   pitch (-HPR): -1.00
t: 1.58,        pitch (+HPR): 163.36,   pitch (-HPR): 394.03
t: 1.67,        pitch (+HPR): 163.34,   pitch (-HPR): -1.00
t: 1.76,        pitch (+HPR): 163.28,   pitch (-HPR): 152.21
t: 1.86,        pitch (+HPR): 163.52,   pitch (-HPR): 183.38
t: 1.95,        pitch (+HPR): 163.49,   pitch (-HPR): -1.00
t: 2.04,        pitch (+HPR): 163.86,   pitch (-HPR): -1.00
t: 2.14,        pitch (+HPR): 163.57,   pitch (-HPR): -1.00
t: 2.23,        pitch (+HPR): 163.46,   pitch (-HPR): 854.43
t: 2.32,        pitch (+HPR): 163.46,   pitch (-HPR): -1.00
t: 2.41,        pitch (+HPR): 163.08,   pitch (-HPR): -1.00
```

### Pre-processing for beat tracking

The [beat-tracking](./beat-tracking) demo applies real-time percussive separation with a hop size of 256 (optimal for percussive results), and applies BTrack on a stream of 256-sized chunks from a wav file.

The clip is the first 10 seconds of Periphery - The Bad Thing. **Verdict:** :x:
![beat-annotated](./docs/annotated_beats.png)
