#ifndef CUDA_MEMORY_HPP
#define CUDA_MEMORY_HPP

#include <cstddef>
#include <cstring>
#include <ostream>

#include "error.hpp"

namespace cuda {
    template <class T>
    class managed_ptr {
    public:
        using element_type = typename std::shared_ptr<T>::element_type;

        managed_ptr() = default;
        managed_ptr(std::size_t count) {
            assert(count > 0);

            T* temp = nullptr;
            CHECK_CUDA(cudaMalloc(&temp, count * sizeof(element_type)));
            wrapped.reset(temp, [](auto ptr) {
                if(ptr != nullptr)
                    CHECK_CUDA(cudaFree(ptr));
            });
        }

        void swap(managed_ptr& other) { wrapped.swap(other.wrapped); }

        void reset() noexcept { wrapped.reset(); }
        void reset(std::size_t count) { managed_ptr(count).swap(*this); }

        long use_count() const noexcept { return wrapped.use_count(); }

        element_type* get() const noexcept {
            return wrapped.get();
        }

        element_type& operator[](std::ptrdiff_t idx) const noexcept {
            return get()[idx];
        }

        element_type& operator*() const noexcept {
            return *get();
        }

        element_type* operator->() const noexcept {
            return get();
        }

        explicit operator bool() const noexcept { return static_cast<bool>(wrapped); }

        template <class T>
        friend bool operator==(const managed_ptr<T>& lhs, const managed_ptr<T>& rhs) noexcept;

        template <class T>
        friend bool operator<(const managed_ptr<T>& lhs, const managed_ptr<T>& rhs) noexcept;

        template <class T, class U, class V>
        friend std::basic_ostream<U, V>& operator<<(std::basic_ostream<U, V>& stream, const managed_ptr<T>& ptr);

    protected:
        std::shared_ptr<T> wrapped;
    };

    template <class T>
    bool operator==(const managed_ptr<T>& lhs, const managed_ptr<T>& rhs) noexcept { return lhs.wrapped == rhs.wrapped; }

    template <class T>
    bool operator!=(const managed_ptr<T>& lhs, const managed_ptr<T>& rhs) noexcept { return !operator==(lhs, rhs); }

    template <class T>
    bool operator<(const managed_ptr<T>& lhs, const managed_ptr<T>& rhs) noexcept { return lhs.wrapped < rhs.wrapped; }

    template <class T>
    bool operator>(const managed_ptr<T>& lhs, const managed_ptr<T>& rhs) noexcept { return operator<(rhs, lhs); }

    template <class T>
    bool operator<=(const managed_ptr<T>& lhs, const managed_ptr<T>& rhs) noexcept { return !operator>(lhs, rhs); }

    template <class T>
    bool operator>=(const managed_ptr<T>& lhs, const managed_ptr<T>& rhs) noexcept { return !operator<(lhs, rhs); }

    /* TODO std::nullptr_t overloads? */

    template <class T, class U, class V>
    std::basic_ostream<U, V>& operator<<(std::basic_ostream<U, V>& stream, const managed_ptr<T>& ptr) {
        stream << ptr.wrapped;
        return stream;
    }

    template< class T >
    void swap(managed_ptr<T>& lhs, managed_ptr<T>& rhs) noexcept { lhs.swap(rhs); }

    template <class T>
    void memcpy(void *dest, const managed_ptr<T>& src, std::size_t n) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpy(dest, src.get(), n, cudaMemcpyDeviceToHost));
    }

    template <class T>
    void memcpy(const managed_ptr<T>& dest, void* src, std::size_t n) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpy(dest.get(), src, n, cudaMemcpyHostToDevice));
    }

    template <class T>
    void memcpy(const managed_ptr<T>& dest, const managed_ptr<T>& src, std::size_t n) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpy(dest.get(), src.get(), n, cudaMemcpyDeviceToDevice));
    }

    template <class T>
    void memset(const managed_ptr<T>& dest, int ch, std::size_t n) {
        assert(n > 0);
        CHECK_CUDA(cudaMemset(dest.get(), ch, n));
    }
}

#endif /* CUDA_MEMORY_HPP */