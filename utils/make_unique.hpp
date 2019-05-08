#ifndef UTILS_MAKE_UNIQUE_HPP
#define UTILS_MAKE_UNIQUE_HPP

#include <memory>
#include <utility>

template<class T, class ...Args>
std::unique_ptr<T> make_unique(Args&& ...args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

#endif /* UTILS_MAKE_UNIQUE_HPP */