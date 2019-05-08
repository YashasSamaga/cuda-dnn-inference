#ifndef CUDA_TENSOR_HPP
#define CUDA_TENSOR_HPP

#include <cuda_runtime.h>

#include "cublas.hpp"
#include "data.hpp"
#include "error.hpp"

namespace cuda {
    template <class T>
    class tensor {
        static_assert(std::is_standard_layout<T>::value, "T must staisfy StandardLayoutType");

    public:
        using value_type = typename common_data<T>::value_type;
        using const_value_type = typename common_data<T>::const_value_type;
        using reference = typename common_data<T>::reference;
        using const_reference = typename common_data<T>::const_reference;
  
        using device_pointer = common_data<value_type>::device_pointer;
        using const_device_pointer = common_data<value_type>::const_device_pointer;
        using host_pointer = common_data<value_type>::host_pointer;
        using const_host_pointer = common_data<value_type>::const_host_pointer;

        using iterator = host_pointer;
        using const_iterator = const_host_pointer;

        tensor() noexcept : batch_size{ 0 }, num_chans{ 0 }, width{ 0 }, height{ 0 } { }
        tensor(stream s) noexcept : data(std::move(s)) { }
        /*
        Add stream overlods TODO
        tensor(int N, int chans, int width, int height) :
            batch_size{ N }, num_chans{ chans }, width{ width }, height{ height }, data(N*chans*width*height) { }
        tensor(int chans, int width, int height) :
            batch_size{ 1 }, num_chans { chans }, width{ width }, height{ height }, data(chans*width*height) { }
        tensor(int width, int height) :
            batch_size{ 1 }, num_chans{ 1 }, width{ width }, height{ height }, data(width*height) { }
        tensor(int height) :
            batch_size{ 1 }, num_chans{ 1 }, width{ 1 }, height{ height }, data(height) { }*/

        auto get_height() const noexcept { return height; }
        auto get_width() const noexcept { return width; }
        auto get_channels() const noexcept { return num_chans; }
        auto get_num_samples() const noexcept { return batch_size; }

        value_type read(int n, int c, int x, int y) const {
            auto idx = n *(num_chans * width * height) + c * (width * height) + y * width + x;
            return data.get_host_readonly()[idx];
        }

        value_type read(int y) const { return read(0, 0, 0, y); }
        value_type read(int x, int y) const { return read(0, 0, x, y); }
        value_type read(int c, int x, int y) const { return read(0, c, x, y); }

        void write(int n, int c, int x, int y, value_type val) {
            auto idx = n * (num_chans * width * height) + c * (width * height) + y * width + x;
            data.get_host_writeable()[idx] = val;
        }

        void write(int y, value_type val) { return write(0, 0, 0, y, val); }
        void write(int x, int y, value_type val) { return write(0, 0, x, y, val); }
        void write(int c, int x, int y, value_type val) { return write(0, c, x, y, val); }

        void resize(std::size_t N, std::size_t chans, std::size_t width, std::size_t height) {
            assert(N != 0 && chans != 0 && width != 0 && height != 0);

            batch_size = N;
            num_chans = chans;
            this->width = width;
            this->height = height;
            data.reset(N * chans * width * height);
        }

        host_pointer get_host_writeable() { return data.get_host_writable(); }
        const_host_pointer get_host_readonly() const { return data.get_host_readonly(); }
        device_pointer get_device_writeable() { return data.get_device_writeable(); }
        const_device_pointer get_device_readonly() const { return data.get_device_readonly(); }

        void copy_to_host() const { data.copy_to_host(); }
        void copy_to_device() const { data.copy_to_device(); }

        iterator begin() { return data.begin(); }
        iterator end() { return data.end(); }
        const_iterator begin() const { return data.begin(); }
        const_iterator end() const { return data.end(); }

    private:
        mutable common_data<value_type> data;
        std::size_t batch_size, num_chans, width, height;
    };

    template <class T>
    void multiply(cublas_context& handle, const tensor<T>& lhs, const tensor<T>& rhs, tensor<T>& result) {
        assert(lhs.get_height() == result.get_height()); //TODO exception
        assert(lhs.get_width() == rhs.get_height());
        assert(rhs.get_width() == result.get_width());

        const auto dest_nr = result.get_height();
        const auto dest_nc = result.get_width();
        const auto lhs_nc = lhs.get_width();
        const auto rhs_nr = rhs.get_height();
        const auto rhs_nc = rhs.get_width();

        blas::gemm<T>(handle, false, false, dest_nc, dest_nr, rhs_nr, 1.0,
            lhs.get_device_readonly(), lhs_nc, rhs.get_device_readonly(), rhs_nc, 0.0,
            result.get_device_writeable(), dest_nc);
    }

    template <class T>
    void add(cublas_context& handle, const tensor<T>& lhs, const tensor<T>& rhs, tensor<T>& result) {
        assert(lhs.get_width() == rhs.get_width());
        assert(lhs.get_height() == rhs.get_height());
        assert(result.get_width() == lhs.get_width());
        assert(result.get_height() == lhs.get_height());
        
        const auto dest_nr = result.get_height();
        const auto dest_nc = result.get_width();
        const auto lhs_nr = lhs.get_height();
        const auto rhs_nr = rhs.get_height();
        const auto rhs_nc = rhs.get_width();

        blas::geam<T>(handle, false, false, dest_nr, dest_nc, 1.0,
            lhs.get_device_readonly(), lhs_nr, 0.0, rhs.get_device_readonly(), rhs_nr,
            result.get_device_writeable(), dest_nr);
    }
}

#endif /* CUDA_TENSOR_HPP */