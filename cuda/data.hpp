#ifndef CUDA_DATA_HPP
#define CUDA_DATA_HPP

#include <cuda_runtime.h>
#include "error.hpp"
#include "../utils/noncopyable.hpp"

namespace cuda {
    template <class T>
    class common_data : noncopyable {
        static_assert(std::is_standard_layout<T>::value, "T must be StandardLayoutType");

    public:
        using value_type = typename std::remove_cv<T>::type;
        using const_value_type = typename std::add_const<value_type>::type;
        using host_pointer = typename std::add_pointer<value_type>::type;
        using const_host_pointer = typename std::add_pointer<const_value_type>::type;
        using device_pointer = typename managed_ptr<T>::pointer;
        using const_device_pointer = typename managed_ptr<T>::const_pointer;

        common_data() : host_dirty{ false }, device_dirty{ false } { str = std::make_shared<stream>(); };
        common_data(std::shared_ptr<stream> s) : str{ std::move(s) } { }
        common_data(std::shared_ptr<stream> s, std::size_t count) : str{ std::move(s) } { reset(count); }
        common_data(std::size_t count) { reset(count); }

        common_data& operator=(common_data&& other) noexcept = default;

        auto begin() {
            host_dirty = true;
            return host_ptr.get();
        }

        auto end() const { return host_ptr.get() + size(); }

        const auto cbegin() const { return host_ptr.get(); }
        const auto cend() const { return host_ptr.get() + size(); }

        auto size() const { return device_ptr.size(); }

        void reset() noexcept { 
            device_ptr.reset();
            host_ptr.reset();
        }

        void reset(std::size_t count) {
            auto d_tmp = managed_ptr<T>(count);
            auto h_tmp = std::shared_ptr<T>(
                pinned_allocator<T>().allocate(count),
                [](auto ptr) { pinned_allocator<T>().deallocate(ptr, 0); });

            device_ptr = std::move(d_tmp);
            host_ptr = std::move(h_tmp);
        }
        
        host_pointer get_host_writeable() {
            host_dirty = true;
            return host_ptr.get();
        }
        
        const_host_pointer get_host_readonly() const {
            return host_ptr.get();
        }

        device_pointer get_device_writeable() {
            device_dirty = true;
            return device_ptr.get();
        }

        const_device_pointer get_device_readonly() const {
            return device_ptr.get();
        }       

        void copy_to_device() const {
           if(host_dirty)
               memcpy((managed_ptr<T>)device_ptr, host_ptr.get(), *str);
        }
        
        void copy_to_host() const { 
            if(device_dirty)
                memcpy(host_ptr.get(), (managed_ptr<T>)device_ptr, *str);
        }

        void synchronize() const { str->synchronize(); }

        bool is_host_dirty() const { return host_dirty; }
        bool is_device_dirty() const { return device_dirty; }

        friend void swap(common_data& lhs, common_data& rhs) {
            using std::swap;
            swap(lhs.str, rhs.str);
            swap(lhs.device_ptr, rhs.device_ptr);
            swap(lhs.device_dirty, rhs.device_dirty);

            swap(lhs.host_ptr, rhs.host_ptr);
            swap(lhs.host_dirty, rhs.device_dirty);
        }

    private:
        std::shared_ptr<stream> str;
        std::shared_ptr<T> host_ptr;
        managed_ptr<T> device_ptr;
        bool host_dirty, device_dirty;
    };
}

#endif /* CUDA_DATA_HPP */