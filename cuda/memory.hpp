#ifndef CUDA_MEMORY_HPP
#define CUDA_MEMORY_HPP

#include <cstddef>
#include <string>
#include <ostream>
#include <type_traits>
#include <cassert>

#include "error.hpp"
#include "pointer.hpp"
#include "span.hpp"
#include "stream.hpp"

namespace cuda {
    /** @brief provides a smart device pointer and allocation/deallocation routines

    The #managed_ptr is a smart device pointer which provides methods to automatically
    allocate and deallocate memory.

    The cv qualifications of T are ignored. It does not make sense to allocate immutable
    or volatile memory and managed_ptr does not provide methods to directly take a pointer.
    
    TODO: handle const qualifications

    @sa device_ptr
    */
    template <class T>
    class managed_ptr {
        static_assert(std::is_standard_layout<T>::value, "T must satisfy StandardLayoutType");

    public:
        using value_type = typename std::remove_cv<typename std::remove_extent<T>::type>::type;
        using size_type = std::size_t;
        using pointer = device_ptr<value_type>;
        using const_pointer = device_ptr<typename std::add_const<value_type>::type>;
        using difference_type = typename pointer::difference_type;

        managed_ptr() noexcept : wrapped{ nullptr }, n{ 0 } { }
        managed_ptr(const managed_ptr&) noexcept = default;
        managed_ptr(managed_ptr&& other) noexcept : wrapped{ std::move(other.wrapped) }, n{ other.n } {
            other.reset();
        }

        managed_ptr(std::size_t count) {
            assert(count > 0);

            auto temp = pointer::pointer(nullptr);
            CHECK_CUDA(cudaMalloc(&temp, count * sizeof(value_type)));
            wrapped.reset(temp, [](auto ptr) {
                if (ptr != nullptr) {
                    CHECK_CUDA(cudaFree(ptr));
                }
                });
            n = count;
        }

        managed_ptr& operator=(managed_ptr&& other) noexcept {
            wrapped = std::move(other.wrapped);
            n = other.n;
            other.reset();
            return *this;
        }

        size_type size() const noexcept { return n; }
        size_type use_count() const noexcept { return wrapped.use_count(); }

        void reset() noexcept { wrapped.reset(); n = 0; }
        void reset(std::size_t count) {
            managed_ptr tmp(count);
            swap(tmp, *this);
        }

        pointer get() const noexcept { return pointer(wrapped.get()); }

        explicit operator bool() const noexcept { return wrapped; }
        operator span<value_type>() const noexcept { return span<value_type>(pointer(wrapped.get()), n); }
        operator span<const value_type>() const noexcept { return span<const value_type>(pointer(wrapped.get()), n); }

        friend bool operator==(const managed_ptr& lhs, const managed_ptr& rhs) noexcept { return lhs.ptr == rhs.ptr; }
        friend bool operator!=(const managed_ptr& lhs, const managed_ptr& rhs) noexcept { return lhs.ptr != rhs.ptr; }

        friend void swap(managed_ptr& lhs, managed_ptr& rhs) noexcept {
            using std::swap;
            swap(lhs.wrapped, rhs.wrapped);
            swap(lhs.n, rhs.n);
        }

        template <class U, class V>
        friend std::basic_ostream<U, V>& operator<<(std::basic_ostream<U, V>& os, const managed_ptr& other) {
            os << other.wrapped << " (device, managed, size: " << other.n * sizeof(T) << " bytes)";
            return os;
        }

    private:
        std::shared_ptr<T> wrapped;
        std::size_t n;
    };

    template <class T>
    void memcpy(T *dest, const managed_ptr<T>& src) {
        memcpy<T>(dest, src.get(), src.size());
    }

    template <class T>
    void memcpy(const managed_ptr<T>& dest, const T* src) {
        memcpy<T>(dest.get(), src, dest.size());
    }

    template <class T>
    void memcpy(const managed_ptr<T>& dest, const managed_ptr<T>& src) {
        memcpy<T>(dest.get(), src.get(), dest.size());
    }

    template <class T>
    void memset(const managed_ptr<T>& dest, int ch) {
        memset<T>(dest.get(), ch, dest.size());
    }

    template <class T>
    void memcpy(T *dest, const managed_ptr<T>& src, const stream& str) {
        memcpy<T>(dest, src.get(), src.size(), str);
    }

    template <class T>
    void memcpy(const managed_ptr<T>& dest, const T* src, const stream& str) {
        memcpy<T>(dest.get(), src, dest.size(), str);
    }

    template <class T>
    void memcpy(managed_ptr<T>& dest, const managed_ptr<T>& src, const stream& str) {
        memcpy<T>(dest.get(), src.get(), dest.size(), str);
    }

    template <class T>
    void memset(const managed_ptr<T>& dest, int ch, const stream& str) {
        memset<T>(dest.get(), ch, dest.size(), str);
    }

    /** @brief provides page-locked host memory for allocator-aware containers
    */
    template <class T>
    struct pinned_allocator {
        using value_type = T;
        using size_type = std::size_t;
        using propogate_on_container_move_assignment = std::true_type; /* pass on noexcept? */

        constexpr pinned_allocator() noexcept = default;
        constexpr pinned_allocator(const pinned_allocator&) noexcept = default;
        template<class U> constexpr pinned_allocator(const pinned_allocator<U>&) noexcept { }

        T* allocate(size_type n) {
            T* ptr;
            try {
                CHECK_CUDA(cudaMallocHost(&ptr, n * sizeof(T)));
            }
            catch (cuda_exception) {
                throw std::bad_alloc(); /* TODO */
            }
            return ptr;
        }

        void deallocate(T* ptr, size_type n) {
            CHECK_CUDA(cudaFreeHost(ptr));
        }
    };

    template<class T1, class T2> inline
    bool operator==(const pinned_allocator<T1>&, const pinned_allocator<T2>&) noexcept {
        return true;
    }

    template<class T1, class T2> inline
    bool operator!=(const pinned_allocator<T1>& lhs, const pinned_allocator<T2>& rhs) noexcept {
        return !(lhs == rhs);
    }
}

#endif /* CUDA_MEMORY_HPP */