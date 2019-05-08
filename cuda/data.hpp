#ifndef CUDA_DATA_HPP
#define CUDA_DATA_HPP

#include "error.hpp"
#include "memory.hpp"
#include "utils/noncopyable.hpp"

#include <type_traits>
#include <memory>

namespace cuda {
    template <class T>
    class gpu_data : noncopyable {
        static_assert(std::is_standard_layout<T>::value, "T must be StandardLayoutType");

    public:
        using value_type = typename std::remove_cv<T>::type;
        using const_value_type = typename std::add_const<value_type>::type;
        using reference = typename std::add_lvalue_reference<value_type>::type;
        using const_reference = typename std::add_lvalue_reference<const_value_type>::type;

        using host_pointer = typename std::add_pointer<value_type>::type;
        using const_host_pointer = typename std::add_pointer<const_value_type>::type;
        using device_pointer = typename managed_ptr<T>::pointer;
        using const_device_pointer = typename managed_ptr<T>::const_pointer;

        gpu_data() : strm(default_stream) { };
        gpu_data(gpu_data&&) noexcept = default;
        gpu_data(stream s) noexcept : strm(std::move(s)) { }
        gpu_data(stream s, std::size_t count) : strm(std::move(s)) { reset(count); }
        gpu_data(std::size_t count) : strm(default_stream) { reset(count); }

        gpu_data& operator=(gpu_data&&) noexcept = default;

        auto size() const noexcept { return device_ptr.size(); }

        void reset() noexcept { device_ptr.reset(); }
        void reset(std::size_t count) { device_ptr = managed_ptr<value_type>(count); }

        device_pointer get_device_writeable() noexcept { return device_ptr.get(); }
        const_device_pointer get_device_readonly() const noexcept { return device_ptr.get(); }

        void copy_to_device(const_host_pointer ptr) { memcpy(device_ptr, ptr, strm); }
        void copy_to_host(host_pointer ptr) const { memcpy(ptr, device_ptr, strm); }

        void synchronize() const { strm.synchronize(); }

        friend void swap(gpu_data& lhs, gpu_data& rhs) noexcept {
            using std::swap;
            swap(lhs.str, rhs.str);
            swap(lhs.device_ptr, rhs.device_ptr);
        }

    private:
        stream strm;
        managed_ptr<value_type> device_ptr;
    };

    template <class T>
    class common_data : noncopyable {
        static_assert(std::is_standard_layout<T>::value, "T must be StandardLayoutType");

    public:
        using value_type = typename std::remove_cv<T>::type;
        using const_value_type = typename std::add_const<value_type>::type;
        using reference = typename std::add_lvalue_reference<value_type>::type;
        using const_reference = typename std::add_lvalue_reference<const_value_type>::type;

        using host_pointer = typename std::add_pointer<value_type>::type;
        using const_host_pointer = typename std::add_pointer<const_value_type>::type;
        using device_pointer = typename managed_ptr<T>::pointer;
        using const_device_pointer = typename managed_ptr<T>::const_pointer;

        common_data() : strm(default_stream_t()), host_dirty{ false }, device_dirty{ false } { };
        common_data(common_data&&) noexcept = default;
        common_data(stream s) noexcept : strm{ std::move(s) } { }
        common_data(stream s, std::size_t count) : strm{ std::move(s) } { reset(count); }
        common_data(std::size_t count) : strm(default_stream) { reset(count); }

        common_data& operator=(common_data&& other) noexcept {
            host_dirty = other.host_dirty;
            device_dirty = other.device_dirty;
            host_ptr = std::move(other.host_ptr);
            device_ptr = std::move(other.device_ptr);
            /* do not move stream */
            return *this;
        }

        host_pointer begin() noexcept { return get_host_writeable(); }
        host_pointer end() noexcept { return get_host_writeable() + size(); }

        const_host_pointer begin() const noexcept { return get_host_readonly(); }
        const_host_pointer end() const noexcept { return get_host_readonly() + size(); }

        const_host_pointer cbegin() const noexcept { return get_host_readonly(); }
        const_host_pointer cend() const noexcept { return get_host_readonly() + size(); }

        auto size() const noexcept { return device_ptr.size(); }

        void reset() noexcept { 
            device_ptr.reset();
            host_ptr.reset();
        }

        void reset(std::size_t count) {
            auto d_tmp = managed_ptr<value_type>(count);

            using allocator = pinned_allocator<value_type>;
            static_assert(std::allocator_traits<allocator>::is_always_equal::value, "");
            auto h_tmp = std::shared_ptr<value_type>(allocator().allocate(count),
                                            [](auto ptr) { allocator().deallocate(ptr, 0); });

            device_ptr = std::move(d_tmp);
            host_ptr = std::move(h_tmp);

            host_dirty = false;
            device_dirty = false;
        }
        
        host_pointer get_host_writeable() noexcept {
            host_dirty = true;
            return host_ptr.get();
        }
        
        const_host_pointer get_host_readonly() const noexcept {
            if (device_dirty) {
                copy_to_host();
                synchronize();
            }
            return host_ptr.get();
        }

        device_pointer get_device_writeable() noexcept {
            device_dirty = true;
            return device_ptr.get();
        }

        const_device_pointer get_device_readonly() const noexcept {
            if (host_dirty) {
                copy_to_device();
                /* synchronization automatically taken care by stream API */
            }
            return device_ptr.get();
        }

        void copy_to_device() const {
            if (host_dirty) {
                host_dirty = false;
                memcpy<value_type>(device_ptr, host_ptr.get(), strm);
            }
        }
        
        void copy_to_host() const { 
            if (device_dirty) {
                device_dirty = false;
                memcpy<value_type>(host_ptr.get(), device_ptr, strm);
            }
        }

        void synchronize() const noexcept { strm.synchronize(); }

        bool is_host_dirty() const noexcept { return host_dirty; }
        bool is_device_dirty() const noexcept { return device_dirty; }

        friend void swap(common_data& lhs, common_data& rhs) noexcept {
            using std::swap;
            swap(lhs.str, rhs.str);
            swap(lhs.device_ptr, rhs.device_ptr);
            swap(lhs.device_dirty, rhs.device_dirty);

            swap(lhs.host_ptr, rhs.host_ptr);
            swap(lhs.host_dirty, rhs.device_dirty);
        }

    private:
        stream strm;
        std::shared_ptr<value_type> host_ptr;
        managed_ptr<value_type> device_ptr;
        mutable bool host_dirty, device_dirty;
    };
}

#endif /* CUDA_DATA_HPP */