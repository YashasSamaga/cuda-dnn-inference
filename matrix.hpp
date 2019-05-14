#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>
#include <type_traits>
#include <memory>
#include <array>
#include <cassert>

template <class T>
class matrix { /* technically, a 4d tensor but let's call it a matrix as it's easier to separate gpu tensor from cpu matrix */
public:
    using value_type = T;
    using const_value_type = typename std::add_const<T>::type;
    using size_type = std::size_t;
    using index_type = size_type;

    matrix() noexcept : sizes{ 0, 0, 0, 0 }, steps{ 0, 0, 0, 0 } { }
    matrix(const matrix&) = default;
    matrix(matrix&&) = default;
    matrix(size_type width) { resize(width); }
    matrix(size_type height, size_type width) { resize(height, width); }
    matrix(size_type chans, size_type height, size_type width) { resize(chans, height, width); }
    matrix(size_type n, size_type chans, size_type height, size_type width) { resize(n, chans, height, width); }

    size_type size() const noexcept { return sizes[0] * sizes[1] * sizes[2] * sizes[3]; }
    size_type get_n() const noexcept { return sizes[count]; }
    size_type get_chans() const noexcept { return sizes[channel]; }
    size_type get_height() const noexcept { return sizes[row]; }
    size_type get_width() const noexcept { return sizes[column]; }

    /* shallow copy; use clone() for deep copy */
    matrix& operator=(matrix&) = default;
    matrix& operator=(matrix&&) = default;

    /* strong exception guarantee */
    void resize(size_type n, size_type chans, size_type height, size_type width) {
        auto size = n * chans * height * width;
        if (size == this->size()) {
            reshape(n, chans, height, width);
            return;
        }

        data = std::shared_ptr<T>(new T[size], [](T* ptr) { delete[] ptr; });

        sizes[count] = n;
        sizes[channel] = chans;
        sizes[row] = height;
        sizes[column] = width;

        /* we store in row major format in memory */
        steps[count] = chans * height * width;
        steps[channel] = height * width;
        steps[row] = width;
        steps[column] = 1;
    }
    void resize(size_type chans, size_type height, size_type width) { resize(1, chans, height, width); }
    void resize(size_type height, size_type width) { resize(1, 1, height, width); }
    void resize(size_type width) { resize(1, 1, 1, width); }

    /* deep copy; strong exception guarantee */
    matrix clone() {
        auto other = matrix(sizes[count], sizes[channel], sizes[row], sizes[column]);
        std::copy_n(data.get(), other.size(), other.get());
        return other;
    }

    /* changes shape without touching the data */
    /* TODO: -1 => figure out the size for the marked axis */
    void reshape(size_type n, size_type chans, size_type height, size_type width) noexcept {
        const auto new_size = n * chans * height * width;
        const auto old_size = size();
        assert(new_size == old_size);

        sizes[count] = n;
        sizes[channel] = chans;
        sizes[row] = height;
        sizes[column] = width;

        /* we store in row major format in memory */
        steps[count] = chans * height * width;
        steps[channel] = height * width;
        steps[row] = width;
        steps[column] = 1;
    }

    value_type& at(index_type n, index_type c, index_type y, index_type x) {
        auto idx = n * steps[count] + c * steps[channel] + y * steps[row] + x * steps[column];
        return data.get()[idx];
    }
    value_type& at(index_type c, index_type y, index_type x) { return at(0, c, y, x); }
    value_type& at(index_type y, index_type x) { return at(0, y, x); }
    value_type& at(index_type x) { return at(0, x); }

    /* TODO add const members */
    const_value_type& at(index_type n, index_type c, index_type y, index_type x) const {
        auto idx = n * steps[count] + c * steps[channel] + y * steps[row] + x * steps[column];
        return data.get()[idx];
    }
    const_value_type& at(index_type c, index_type y, index_type x) const { return at(0, c, y, x); }
    const_value_type& at(index_type y, index_type x) const { return at(0, y, x); }
    const_value_type& at(index_type x) const { return at(0, x); }

    friend void swap(matrix& lhs, matrix& rhs) noexcept {
        using std::swap;
        swap(lhs.data, rhs.data);
        swap(lhs.steps, rhs.steps);
        swap(lhs.sizes, rhs.sizes);
    }

private:
    enum {
        count,
        channel,
        row,
        column
    };

    std::array<size_type, 4> steps;
    std::array<size_type, 4> sizes;
    std::shared_ptr<T> data;
};

#endif /* MATRIX_HPP */