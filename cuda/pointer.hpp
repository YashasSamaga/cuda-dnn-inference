#ifndef CUDA_POINTER_HPP
#define CUDA_POINTER_HPP

#include "error.hpp"
#include "stream.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>
#include <ostream>
#include <cassert>

namespace cuda {
    /** @brief provides type-safe device pointer

        The #device_ptr warps a raw pointer and does not implicitly convert to raw pointers.
        This ensures that accidental mixing of host and device pointers do not happen.

        It is meant to point to locations in device memory. Hence, it provides dereferencing or
        array subscript capability for device code only.

        A const device_ptr<T> can represents an immutable pointer to a mutable memory.
        A device_ptr<const T> represents a mutable pointer to an immutable memory.
        A const device_ptr<const T> represents an immutable pointer to an immutable memory.

        A device_ptr<T> can implicitly convert to device_ptr<const T>.

        @sa managed_ptr
    */
    template <class T>
    class device_ptr {
        static_assert(std::is_standard_layout<T>::value, "T must satisfy StandardLayoutType");

    public:
        using element_type = typename std::remove_extent<T>::type;
        using difference_type = std::ptrdiff_t;
        using pointer = typename std::add_pointer<element_type>::type;
        using reference = typename std::add_lvalue_reference<element_type>::type;

        constexpr device_ptr() = default;
        __host__ __device__ constexpr explicit device_ptr(pointer ptr_) noexcept : ptr{ ptr_ } { }

        __host__ __device__ constexpr device_ptr operator=(pointer ptr_) noexcept { ptr = ptr_; return *this; }

        __host__ __device__ constexpr pointer get() const noexcept { return ptr; };
        __device__ constexpr reference operator[](difference_type idx) const noexcept { return get()[idx]; }
        __device__ constexpr reference operator*() const noexcept { return *get(); }
        __device__ constexpr pointer operator->() const noexcept { return get(); }

        template<class U = T, class V = std::add_const<U>::type,
            typename std::enable_if<!std::is_const<U>::value, bool>::type = true>
            __host__ __device__ operator device_ptr<V>() const noexcept { return device_ptr<V>{ptr}; }

        __host__ __device__ constexpr explicit operator bool() const noexcept { return ptr; }

        __host__ __device__ constexpr device_ptr operator++() noexcept {
            ++ptr;
            return *this;
        }
        __host__ __device__ constexpr device_ptr operator++(int) noexcept {
            auto tmp = device_ptr(*this);
            ptr++;
            return tmp;
        }

        __host__ __device__ constexpr device_ptr operator--() noexcept {
            --ptr;
            return *this;
        }

        __host__ __device__ constexpr device_ptr operator--(int) noexcept {
            auto tmp = device_ptr(*this);
            ptr--;
            return tmp;
        }

        __host__ __device__ constexpr device_ptr operator+=(std::ptrdiff_t offset) noexcept {
            ptr += offset;
            return *this;
        }

        __host__ __device__ constexpr device_ptr operator-=(std::ptrdiff_t offset) noexcept {
            ptr -= offset;
            return *this;
        }

        __host__ __device__ friend constexpr device_ptr operator+(device_ptr lhs, std::ptrdiff_t offset) noexcept {
            return lhs += offset;
        }

        __host__ __device__ friend constexpr device_ptr operator-(device_ptr lhs, std::ptrdiff_t offset) noexcept {
            return lhs -= offset;
        }

        __host__ __device__ friend constexpr difference_type operator-(device_ptr lhs, device_ptr rhs) noexcept {
            return lhs.ptr - rhs.ptr;
        }

        __host__ __device__ friend constexpr bool operator==(device_ptr lhs, device_ptr rhs) noexcept { return lhs.ptr == rhs.ptr; }
        __host__ __device__ friend constexpr bool operator!=(device_ptr lhs, device_ptr rhs) noexcept { return !(lhs == rhs); }
        __host__ __device__ friend constexpr bool operator<(device_ptr lhs, device_ptr rhs) noexcept { return lhs.ptr < rhs.ptr; }
        __host__ __device__ friend constexpr bool operator>(device_ptr lhs, device_ptr rhs) noexcept { return rhs < lhs; }
        __host__ __device__ friend constexpr bool operator<=(device_ptr lhs, device_ptr rhs) noexcept { return !(rhs < lhs); }
        __host__ __device__ friend constexpr bool operator>=(device_ptr lhs, device_ptr rhs) noexcept { return !(lhs < rhs); }

        __host__ __device__ explicit operator pointer() const noexcept { return ptr; }

        __host__ friend void swap(device_ptr& lhs, device_ptr& rhs) noexcept {
            /* not device code because std::swap does not work in device
            ** but could a custom implementation of gpu::swap do the job? */
            using std::swap;
            swap(lhs.ptr, rhs.ptr);
        }

        template <class U, class V>
        __host__ friend std::basic_ostream<U, V>& operator<<(std::basic_ostream<U, V>& os, device_ptr other) {
            os << other.get() << " (device)";
            return os;
        }

    private:
        pointer ptr;
    };

    /* default stream */
    template <class T>
    void memcpy(T *dest, const T *src, std::size_t n) {
        assert(n > 0);
        cudaMemcpy(dest, src, n * sizeof(T), cudaMemcpyDefault);
    }

    template <class T>
    void memcpy(T *dest, device_ptr<const T> src, std::size_t n) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpy(dest, src.get(), n * sizeof(T), cudaMemcpyDefault));
    }

    template <class T>
    void memcpy(device_ptr<T> dest, const T* src, std::size_t n) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpy(dest.get(), src, n * sizeof(T), cudaMemcpyDefault));
    }

    template <class T>
    void memcpy(device_ptr<T> dest, device_ptr<const T> src, std::size_t n) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpy(dest.get(), src.get(), n * sizeof(T), cudaMemcpyDefault));
    }

    template <class T>
    void memset(device_ptr<T> dest, int ch, std::size_t n) {
        assert(n > 0);
        assert(ch < 128 && ch >= -128);
        CHECK_CUDA(cudaMemset(dest.get(), ch, n * sizeof(T)));
    }

    /* stream based */
    template <class T> 
    void memcpy(T *dest, device_ptr<const T> src, std::size_t n, const stream& str) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpyAsync(dest, src.get(), n * sizeof(T), cudaMemcpyDefault, str));
    }

    template <class T>
    void memcpy(device_ptr<T> dest, const T *src, std::size_t n, const stream& str) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpyAsync(dest.get(), src, n * sizeof(T), cudaMemcpyDefault, str));
    }

    template <class T>
    void memcpy(device_ptr<T> dest, device_ptr<const T> src, std::size_t n, const stream& str) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpyAsync(dest.get(), src.get(), n * sizeof(T), cudaMemcpyDefault, str));
    }

    template <class T>
    void memset(device_ptr<T> dest, int ch, std::size_t n, const stream& str) {
        assert(n > 0);
        assert(ch < 128 && ch >= -128);
        CHECK_CUDA(cudaMemsetAsync(dest.get(), ch, n * sizeof(T), str));
    }
}

#endif /* CUDA_POINTER_HPP */