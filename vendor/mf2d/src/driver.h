#ifndef DRIVER_H
#define DRIVER_H

template <typename T>
struct Image2D {
    int x;
    int y;
    T* p;

    Image2D() : x(1), y(1), p(0)
    {}

    Image2D(int x_, int y_) : x(x_), y(y_), p(0)
    {}

    void like(Image2D<T> o) {
        x = o.x;
        y = o.y;
    }

    inline int size() const {
        return x * y;
    }

    inline void alloc() {
        p = new T[size()];
    }
};


template <typename T>
struct Image1D {
    int x;
    T* p;

    Image1D() : x(1), p(0)
    {}

    Image1D(int x_) : x(x_), p(0)
    {}

    void like(Image1D<T> o) {
        x = o.x;
    }

    inline int size() const {
        return x;
    }

    inline void alloc() {
        p = new T[size()];
    }
};


class VDriver {
public:
    virtual void process(int h) = 0;
    virtual void diff() = 0;
    virtual void write(const char* filename) = 0;
    virtual void benchmark() = 0;
    virtual ~VDriver() {}
};


template <typename T, typename I>
class Driver : public VDriver
{
public:
    // Driver will own img.p
    Driver(I img) : in(img)
    {
        out.like(in);
        out.alloc();
    }

    ~Driver() {
        delete in.p;
        delete out.p;
    }

    void process(int h);
    void diff();
    void write(const char* filename);
    void benchmark();

private:
    I in;
    I out;
};


#endif
