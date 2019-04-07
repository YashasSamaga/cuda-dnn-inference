#ifndef UTILS_NONCOPYABLE_HPP
#define UTILS_NONCOPYABLE_HPP

class noncopyable {
public:
    noncopyable() = default;

private:
    noncopyable(const noncopyable&);
    noncopyable& operator=(const noncopyable&);
};

#endif /* UTILS_NONCOPYABLE_HPP */