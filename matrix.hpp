#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>
#include <type_traits>
#include <memory>

template <class T>
class matrix {
public:
    using value_type = T;

    matrix() noexcept : rows{ 0 }, cols{ 0 } { }
    matrix(const matrix&) = default;
    matrix(matrix&&) = default;
    matrix(std::size_t cols) { resize(1, cols); }
    matrix(std::size_t rows, std::size_t cols) { resize(rows, cols); }

    std::size_t size() const noexcept { return rows * cols; }
    std::size_t get_rows() const noexcept { return rows; }
    std::size_t get_cols() const noexcept { return cols; }

    matrix& operator=(matrix&) = default;
    matrix& operator=(matrix&&) = default;

    void resize(std::size_t rows, std::size_t cols) {
        this->rows = rows;
        this->cols = cols;
        auto size = rows * cols;
        data = std::shared_ptr<T>(new T[size], [](T* ptr) { delete[] ptr; });
    }

    matrix clone() {
        auto other = matrix(rows, cols);
        std::copy_n(data.get(), other.size(), other.get());
        return other;
    }

    value_type& at(int x, int y) const {
        auto idx = y * cols + x;
        return data.get()[idx];
    }

    value_type& at(int y) const { return at(0, y); }

    friend void swap(matrix& lhs, matrix& rhs) noexcept {
        using std::swap;
        swap(lhs.data, rhs.data);
        swap(lhs.rows, rhs.rows);
        swap(lhs.cols, rhs.cols);
    }

private:
    std::size_t rows, cols;
    std::shared_ptr<T> data;
};

#endif /* MATRIX_HPP */