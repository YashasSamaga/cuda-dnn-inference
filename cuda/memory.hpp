#ifndef CUDA_MEMORY_HPP
#define CUDA_MEMORY_HPP

#include <cstddef>
#include <cstring>
#include <ostream>
#include <type_traits>

#include "error.hpp"

namespace cuda {
	/** @brief provides type-safe device pointer

	The #device_ptr warps a raw pointer and does not implicitly convert to raw pointers.
	This ensures that accidental mixing of host and device pointers do not happen.

	It is meant to point to locations in device memory. Hence, it provides dereferencing or]
	array subscript capability for device code only.

	A const device_ptr<T> can represents an immutable pointer to a mutable memory.
	A device_ptr<const T> represents a mutable pointer to an immutable memory.
	A const device_ptr<const T> represents an immutable pointer to an immutable memory.

	For readability purposes, if the pointer points to an array, you may use device_ptr<T[]> which
	is equivalent to device_ptr<T>.

	@sa managed_ptr
	*/
    template <class T>
    class device_ptr {
        static_assert(std::is_standard_layout<T>::value, "T must satisfy StandardLayoutType");
    public:
		using element_type		= std::remove_extent_t<T>;
		using difference_type	= std::ptrdiff_t;
		using pointer			= std::add_pointer_t<element_type>;
		using reference			= std::add_lvalue_reference_t<element_type>;

		constexpr device_ptr() = default;
		__host__ __device__ constexpr explicit device_ptr(pointer ptr_) noexcept : ptr{ ptr_ } { }

		__host__ __device__ constexpr device_ptr operator=(std::nullptr_t) noexcept { ptr = nullptr; return *this; }
		__host__ __device__ constexpr device_ptr operator=(pointer ptr_) noexcept { ptr = ptr_; return *this; }

		__host__ __device__ constexpr pointer get() const noexcept { return ptr; };
		__device__ constexpr reference operator[](difference_type idx) const noexcept { return get()[idx]; }
		__device__ constexpr reference operator*() const noexcept { return *get(); }
		__device__ constexpr pointer operator->() const noexcept { return get(); }

		__host__ __device__ constexpr explicit operator bool() const noexcept { return ptr; }

		__host__ __device__ constexpr device_ptr operator++() noexcept {
            ++ptr;
            return *this;
        }

		__host__ __device__ constexpr device_ptr operator++(int) noexcept {
            device_ptr tmp(*this);
            ptr++;
            return tmp;
        }

		__host__ __device__ constexpr device_ptr operator--() noexcept {
            --ptr;
            return *this;
        }

		__host__ __device__ constexpr device_ptr operator--(int) noexcept {
            device_ptr tmp(*this);
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
            return lhs + offset;
        }

		__host__ __device__ friend constexpr device_ptr operator-(device_ptr lhs, std::ptrdiff_t offset) noexcept {
            return lhs - offset;
        }

		__host__ __device__ friend constexpr difference_type operator-(device_ptr lhs, device_ptr rhs) noexcept {
            return lhs.ptr - rhs.ptr;
        }

		__host__ __device__ friend constexpr bool operator==(device_ptr lhs, device_ptr rhs) noexcept { return lhs.ptr == rhs.ptr; }
		__host__ __device__ friend constexpr bool operator!=(device_ptr lhs, device_ptr rhs) noexcept { return !(lhs == rhs); }
		__host__ __device__ friend constexpr bool operator<(device_ptr lhs, device_ptr rhs) noexcept { return lhs.ptr < rhs.ptr; }
		__host__ __device__ friend constexpr bool operator>(device_ptr lhs, device_ptr rhs)  { return rhs < lhs; }
		__host__ __device__ friend constexpr bool operator<=(device_ptr lhs, device_ptr rhs) { return !(lhs > rhs); }
		__host__ __device__ friend constexpr bool operator>=(device_ptr lhs, device_ptr rhs) { return !(lhs < rhs); }

		__host__ friend void swap(device_ptr& lhs, device_ptr& rhs) noexcept {
			using std::swap;
			swap(lhs.ptr, rhs.ptr);
		}

        template <class U, class V>
        __host__ friend std::basic_ostream<U, V>& operator<<(std::basic_ostream<U, V>& os, const device_ptr other) {
            os << other.get() << " (device)";
            return os;
        }

    private:
        pointer ptr;
    };

	/** @brief provides non-owning mutable view for device arrays

		@note does not update smart pointers
	*/
	template <class T>
	class span {
		static_assert(std::is_standard_layout<T>::value, "T must satisfy StandardLayoutType");
	public:
		using value_type = std::remove_cv_t<T>;
		using element_type = T;
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;
		using pointer = device_ptr<value_type>;
		using const_pointer = std::add_const_t<device_ptr<value_type>>;
		using reference = std::add_lvalue_reference_t<value_type>;
		using const_reference = std::add_lvalue_reference_t<std::add_const_t<value_type>>;

		constexpr span() noexcept = default;
		__host__ __device__ constexpr span(pointer first, pointer last) noexcept : ptr{ first }, sz{ last - first } { }
		__host__ __device__ constexpr span(pointer first, size_type count) noexcept : ptr{ first }, sz{ count } { }

		__device__ constexpr reference operator[](difference_type index) { return ptr[index]; }
		__device__ constexpr const_reference operator[](difference_type index) const { return ptr[index]; }

		__host__ __device__ pointer data() noexcept { return ptr; }
		__host__ __device__ const_pointer data() const noexcept { return ptr; }

		__host__ __device__ size_type size() const noexcept { return sz; }

	private:
		pointer ptr;
		std::size_t sz;
	};

    template <class T = unsigned char>
    void memcpy(void *dest, const void *src, std::size_t n) {
        assert(n > 0);
        cudaMemcpy(dest, src, n * sizeof(T), cudaMemcpyDefault);
    }

    template <class T>
    void memcpy(void *dest, const device_ptr<T> src, std::size_t n) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpy(dest, src.get(), n * sizeof(T), cudaMemcpyDefault));
    }

    template <class T>
    void memcpy(device_ptr<T> dest, const void* src, std::size_t n) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpy(dest.get(), src, n * sizeof(T), cudaMemcpyDefault));
    }

    template <class T>
    void memcpy(device_ptr<T> dest, const device_ptr<T> src, std::size_t n) {
        assert(n > 0);
        CHECK_CUDA(cudaMemcpy(dest.get(), src.get(), n * sizeof(T), cudaMemcpyDefault));
    }

    template <class T>
    void memset(device_ptr<T> dest, int ch, std::size_t n) {
        assert(n > 0);
        assert(ch < 128 && ch >= -128);
        CHECK_CUDA(cudaMemset(dest.get(), ch, n * sizeof(T)));
    }

	/** @brief provides a smart device pointer and allocation/deallocation routines

	The #managed_ptr is a smart device pointer which provides methods to automatically
	allocate and deallocate memory.

	@sa device_ptr
	*/
    template <class T>
    class managed_ptr {
        static_assert(std::is_standard_layout<T>::value, "T must satisfy StandardLayoutType");
    public:
		using value_type		= std::remove_cv_t<std::remove_extent_t<T>>;
		using size_type			= std::size_t;
		using difference_type	= std::ptrdiff_t;
		using pointer			= device_ptr<value_type>;
		using const_pointer		= std::add_const_t<device_ptr<value_type>>;

        managed_ptr() noexcept : wrapped{ nullptr }, n{ 0 } { }
        managed_ptr(const managed_ptr&) noexcept = default;
        managed_ptr(managed_ptr&& other) noexcept : wrapped{ other.ptr }, n{ other.n } {
            other.reset();
        }

        managed_ptr(std::size_t count) {
            assert(count > 0);

            typename pointer::pointer temp = nullptr;
            CHECK_CUDA(cudaMalloc(&temp, count * sizeof(value_type)));
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

		size_type size() const noexcept { return n; }
		size_type use_count() const noexcept { return wrapped.use_count(); }

        void reset() noexcept { wrapped.reset(); n = 0; }
        void reset(std::size_t count) { 
            managed_ptr tmp(count);
            swap(tmp, *this);
        }

        pointer get() noexcept { return pointer(wrapped.get()); }
        const_pointer get() const noexcept { return const_pointer(wrapped.get()); }

        explicit operator bool() const noexcept { return wrapped; }
		operator span<T>() { return span<T>(pointer(wrapped.get()), n); }

        friend bool operator==(const managed_ptr& lhs, const managed_ptr& rhs) noexcept { return lhs.ptr == rhs.ptr; }
		friend bool operator!=(const managed_ptr& lhs, const managed_ptr& rhs) noexcept { return lhs.ptr != rhs.ptr; }

		friend void swap(managed_ptr& lhs, managed_ptr& rhs) noexcept {
			using std::swap;
			std::swap(lhs.wrapped, rhs.wrapped);
			std::swap(lhs.n, rhs.n);
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
    void memcpy(void *dest, const managed_ptr<T>& src) {
		memcpy(dest, src.get(), src.size());
    }

    template <class T>
    void memcpy(managed_ptr<T>& dest, const void* src) {
        memcpy(dest.get(), src, dest.size());
    }

    template <class T>
    void memcpy(managed_ptr<T>& dest, const managed_ptr<T>& src) {
        assert(n > 0);
		memcpy(dest.get(), src.get(), dest.size());
    }

    template <class T>
    void memset(managed_ptr<T>& dest, int ch, std::size_t n) {
        memset(dest.get(), ch, dest.size());
    }

	/** @brief provides page-locked host memory for allocator-aware containers
	*/
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