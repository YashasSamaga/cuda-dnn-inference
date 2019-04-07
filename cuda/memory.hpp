#ifndef CUDA_MEMORY_HPP
#define CUDA_MEMORY_HPP

#include <cstddef>
#include <cstring>
#include <ostream>
#include <type_traits>

#include "error.hpp"

#include "../utils/operators.hpp"

namespace cuda {
    template <class T>
    class device_raw_ptr : public relational_operators<T> {
        static_assert(std::is_standard_layout<T>::value, "T must satisfy StandardLayoutType");
    public:
        using element_type = std::remove_extent_t<T>;

        constexpr device_raw_ptr() noexcept : ptr{ nullptr } { };
        constexpr device_raw_ptr(const device_raw_ptr& other) noexcept = default;
        explicit device_raw_ptr(element_type* ptr_) noexcept : ptr{ ptr_ } { }
        device_raw_ptr(device_raw_ptr&& other) noexcept : ptr{ other.ptr } { other.reset(); }

        device_raw_ptr& operator=(const device_raw_ptr& other) noexcept { 
            ptr = other.ptr;
            return *this;
        }

        device_raw_ptr& operator=(device_raw_ptr&& other) noexcept { 
            ptr = other.ptr;
            other.reset();
            return *this;
        }

        void reset() noexcept { ptr = nullptr; }
        void reset(element_type* ptr_) noexcept { ptr = ptr_; }

        element_type* get() noexcept { return ptr; };
        const element_type* get() const noexcept { return ptr; }

        friend void swap(device_raw_ptr& lhs, device_raw_ptr& rhs) noexcept {
            using std::swap;
            std::swap(lhs.ptr, rhs.ptr);
        }

        explicit operator bool() const noexcept { return ptr; }

        device_raw_ptr& operator++() noexcept {
            ++ptr;
            return *this;
        }

        device_raw_ptr operator++(int) noexcept {
            device_raw_ptr tmp(*this);
            ptr++;
            return tmp;
        }

        device_raw_ptr& operator--() noexcept {
            --ptr;
            return *this;
        }

        device_raw_ptr operator--(int) noexcept {
            device_raw_ptr tmp(*this);
            ptr--;
            return tmp;
        }

        device_raw_ptr& operator+=(std::ptrdiff_t offset) noexcept {
            ptr += offset;
            return *this;
        }

        device_raw_ptr& operator-=(std::ptrdiff_t offset) noexcept {
            ptr -= offset;
            return *this;
        }
        
        friend device_raw_ptr& operator+(device_raw_ptr lhs, std::ptrdiff_t offset) noexcept {
            lhs += offset;
            return lhs;
        }

        friend device_raw_ptr& operator-(device_raw_ptr lhs, std::ptrdiff_t offset) noexcept {
            lhs -= offset;
            return lhs;
        }

        /* required by relational_operators base class */
        friend bool operator==(const device_raw_ptr& lhs, const device_raw_ptr& rhs) noexcept { return lhs.ptr == rhs.ptr; }
        friend bool operator<(const device_raw_ptr& lhs, const device_raw_ptr& rhs) noexcept { return lhs.ptr < rhs.ptr; }

        template <class U, class V>
        friend std::basic_ostream<U, V>& operator<<(std::basic_ostream<U, V>& os, const device_raw_ptr& other) {
            os << other.get() << " (device)";
            return os;
        }

    private:
        T *ptr;
    };

    template <class T = unsigned char>
    void memcpy(void *dest, const void *src, std::size_t n) {
        assert(n > 0);
        cudaMemcpy(dest, src, n * sizeof(T), cudaMemcpyDefault);
    }

    template <class T>
    void memcpy(void *dest, const device_raw_ptr<T> src, std::size_t n) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpy(dest, src.get(), n * sizeof(T), cudaMemcpyDefault));
    }

    template <class T>
    void memcpy(device_raw_ptr<T> dest, const void* src, std::size_t n) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpy(dest.get(), src, n * sizeof(T), cudaMemcpyDefault));
    }

    template <class T>
    void memcpy(device_raw_ptr<T> dest, const device_raw_ptr<T> src, std::size_t n) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpy(dest.get(), src.get(), n * sizeof(T), cudaMemcpyDefault));
    }

    template <class T>
    void memset(device_raw_ptr<T> dest, int ch, std::size_t n) {
        assert(n > 0);
        assert(ch < 128 && ch >= -128);
        CHECK_CUDA(cudaMemset(dest.get(), ch, n * sizeof(T)));
    }

    template <class T>
    class managed_ptr : public equality_operators<managed_ptr<T>> {
        static_assert(std::is_standard_layout<T>::value, "T must satisfy StandardLayoutType");
    public:
        using element_type = std::remove_extent_t<T>;

        constexpr managed_ptr() noexcept : wrapped{ nullptr }, n{ 0 } { }
        constexpr managed_ptr(const managed_ptr&) noexcept = default;
        managed_ptr(managed_ptr&& other) noexcept : wrapped{ other.ptr }, n{ other.n } {
            other.reset();
        }

        managed_ptr(std::size_t count) {
            assert(count > 0);

            element_type* temp = nullptr;
            CHECK_CUDA(cudaMalloc(&temp, count * sizeof(element_type)));
            wrapped.reset(temp, [](auto ptr) {
                if (ptr != nullptr) {
                    CHECK_CUDA(cudaFree(ptr));
                }
            });
            n = count;
        }

        managed_ptr& operator=(managed_ptr other) noexcept {
            swap(other, *this);
            return *this;
        }

        managed_ptr& operator=(managed_ptr&& other) noexcept {
            wrapped = std::move(other.wrapped);
            n = other.n;
            other.reset();
            return *this;
        }

        std::size_t size() const noexcept { return n; }
        std::size_t use_count() const noexcept { return wrapped.use_count(); }

        void reset() noexcept { wrapped.reset(); n = 0; }
        void reset(std::size_t count) { 
            managed_ptr tmp(count);
            swap(tmp, *this);
        }

        element_type* get() noexcept { return wrapped.get(); }
        const element_type* get() const noexcept { return wrapped.get(); }

        friend void swap(managed_ptr& lhs, managed_ptr& rhs) noexcept { 
            using std::swap;
            std::swap(lhs.wrapped, rhs.wrapped);
            std::swap(lhs.n, rhs.n);
        }

        explicit operator bool() const noexcept { return wrapped; }

        friend bool operator==(const managed_ptr& lhs, const managed_ptr& rhs) noexcept { return lhs.ptr == rhs.ptr; }

        template <class U, class V>
        friend std::basic_ostream<U, V>& operator<<(std::basic_ostream<U, V>& os, const managed_ptr& other) {
            os << other.wrapped << " (device, managed, size: " << other.n * sizeof(T) << " bytes)";
            return os;
        }

    protected:
        std::shared_ptr<T> wrapped;
        std::size_t n;
    };

    template <class T>
    void memcpy(void *dest, const managed_ptr<T>& src) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpy(dest, src.get(), src.size() * sizeof(T), cudaMemcpyDefault));
    }

    template <class T>
    void memcpy(managed_ptr<T>& dest, const void* src) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpy(dest.get(), src, dest.size() * sizeof(T), cudaMemcpyDefault));
    }

    template <class T>
    void memcpy(managed_ptr<T>& dest, const managed_ptr<T>& src) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpy(dest.get(), src.get(), dest.size() * sizeof(T), cudaMemcpyDefault));
    }

    template <class T>
    void memset(managed_ptr<T>& dest, int ch, std::size_t n) {
        assert(n > 0);
        assert(ch < 128 && ch >= -128);
        CHECK_CUDA(cudaMemset(dest.get(), ch, dest.size() * sizeof(T)));
    }

    template <class T>
    struct pinned_allocator {
        using value_type    = T;
        using size_type     = std::size_t;
        using propogate_on_container_move_assignment = std::true_type; /* pass on noexcept? */

        constexpr pinned_allocator() noexcept = default;
        constexpr pinned_allocator(const pinned_allocator&) noexcept = default;
        template<class U> constexpr pinned_allocator(const pinned_allocator<U>&) noexcept { }
    
        T* allocate(size_type n) {
            T* ptr;
            try {
                CHECK_CUDA(cudaMallocHost(&ptr, n * sizeof(T)));
            } catch (cuda_exception) {
                throw std::bad_alloc(); /* TODO */
            }
            return ptr;
        }

        void deallocate(T* ptr, size_type n) {
            CHECK_CUDA(cudaFreeHost(ptr));
        }

        template<class T1, class T2>
        friend bool operator==(const pinned_allocator<T1>&, const pinned_allocator<T2>&) noexcept {
            return true;
        }

        template<class T1, class T2>
        friend bool operator!=(const pinned_allocator<T1>& lhs, const pinned_allocator<T2>& rhs) noexcept {
            return !(lhs == rhs);
        }
    };
}

#endif /* CUDA_MEMORY_HPP */