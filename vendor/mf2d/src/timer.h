#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>

class Timer {
public:
    Timer() : start(get_time())
    {}

    double peek() {
        double now = get_time();
        return now - start;
    }

private:
    static double get_time() {
        struct timeval tm;
        gettimeofday(&tm, NULL);
        return static_cast<double>(tm.tv_sec) + static_cast<double>(tm.tv_usec) / 1E6;
    }

    double start;
};

#endif
