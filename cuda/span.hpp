#ifndef CUDA_SPAN_HPP
#define CUDA_SPAN_HPP

#include "pointer.hpp"

namespace cuda {
    /** @brief provides non-owning mutable view for device arrays

        const span<T>/span<T> provides mutable access to the elements unless T is const qualified
        const span<T> makes the span immutable but not the elements
    */
    template <class T>
    class span {
        static_assert(std::is_standard_layout<T>::value, "T must satisfy StandardLayoutType");

    public:
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using pointer = device_ptr<value_type>;
        using const_pointer = device_ptr<std::add_const<value_type>::type>;
        using reference = typename std::add_lvalue_reference<value_type>::type;
        using const_reference = typename std::add_lvalue_reference<std::add_const<value_type>::type>;
        using iterator = pointer;
        using const_iterator = const_pointer;

        constexpr span() noexcept : ptr{ nullptr }, sz{ 0 } { }
        __host__ __device__ constexpr span(pointer first, pointer last) noexcept : ptr{ first }, sz{ last - first } { }
        __host__ __device__ constexpr span(pointer first, size_type count) noexcept : ptr{ first }, sz{ count } { }

        __host__ __device__ constexpr size_type size() const noexcept { return sz; }
        __host__ __device__ constexpr bool empty() const noexcept { return size() == 0; }

        __device__ constexpr reference operator[](difference_type index) const { return ptr[index]; }
        __host__ __device__ constexpr pointer data() const noexcept { return ptr; }

    private:
        pointer ptr;
        std::size_t sz;
    };

    /** @brief provides non-owning immutable view for device arrays

        @sa span
    */
    template <class T>
    using view = span<const T>;
}

#endif /* CUDA_SPAN_HPP */