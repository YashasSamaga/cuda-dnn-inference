#ifndef CUDA_TENSOR_HPP
#define CUDA_TENSOR_HPP

#include <cuda_runtime.h>

#include "cublas.hpp"
#include "math.cuh"
#include "cudnn.hpp"
#include "data.hpp"
#include "error.hpp"
#include "workspace.hpp"
#include "matrix.hpp"

namespace cuda {
    template <class T>
    class tensor : noncopyable {
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

        using size_type = std::size_t;
        using index_type = size_type;

        tensor() noexcept : batch_size{ 0 }, num_chans{ 0 }, width{ 0 }, height{ 0 } { }
        tensor(tensor&& other) noexcept {
            data = std::move(other.data);
            batch_size = other.batch_size;
            num_chans = other.num_chans;
            height = other.height;
            width = other.width;
            other.batch_size = 0;
            other.num_chans = 0;
            other.height = 0;
            other.width = 0;
        }
        tensor(stream s) noexcept : data(std::move(s)), batch_size{ 0 }, num_chans{ 0 }, width{ 0 }, height{ 0 } { }

        tensor& operator=(tensor&& other) noexcept {
            data = std::move(other.data);
            batch_size = other.batch_size;
            num_chans = other.num_chans;
            height = other.height;
            width = other.width;
            other.batch_size = 0;
            other.num_chans = 0;
            other.height = 0;
            other.width = 0;
            return *this;
        }

        auto size() const noexcept { return batch_size * num_chans * width * height; }
        auto get_height() const noexcept { return height; }
        auto get_width() const noexcept { return width; }
        auto get_chans() const noexcept { return num_chans; }
        auto get_num_samples() const noexcept { return batch_size; }

        value_type read(index_type n, index_type c, index_type y, index_type x) const {
            auto idx = n *(num_chans * width * height) + c * (width * height) + y * width + x;
            return data.get_host_readonly()[idx];
        }
        value_type read(index_type x) const { return read(0, 0, 0, x); }
        value_type read(index_type y, index_type x) const { return read(0, 0, y, x); }
        value_type read(index_type c, index_type y, index_type x) const { return read(0, c, y, x); }

        void write(index_type n, index_type c, index_type y, index_type x, value_type val) {
            auto idx = n * (num_chans * height * width) + c * (width * height) + y * width + x;
            data.get_host_writeable()[idx] = val;
        }
        void write(index_type x, value_type val) { return write(0, 0, 0, x, val); }
        void write(index_type y, index_type x, value_type val) { return write(0, 0, y, x, val); }
        void write(index_type c, index_type y, index_type x, value_type val) { return write(0, c, y, x, val); }

        void resize(size_type n, size_type chans, size_type height, size_type width) {
            assert(n != 0 && chans != 0 && width != 0 && height != 0);
            const auto new_size = n * chans * height * width;
            const auto old_size = size();
            if (new_size == old_size && new_size == data.size()) {
                reshape(n, chans, height, width);
                return;
            }

            data.reset(new_size);
            batch_size = n;
            num_chans = chans;
            this->height = height;
            this->width = width;
        }
        void resize(size_type chans, size_type height, size_type width) { resize(1, chans, height, width); }
        void resize(size_type height, size_type width) { resize(1, 1, height, width); }
        void resize(size_type width) { resize(1, 1, 1, width); }
        
        /* TODO remove const */
        void reshape(size_type n, size_type chans, size_type height, size_type width) const noexcept {
            const auto new_size = n * chans * height * width;
            const auto old_size = size();
            assert(new_size == old_size);

            batch_size = n;
            num_chans = chans;
            this->height = height;
            this->width = width;
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

        friend void swap(tensor& lhs, tensor& rhs) noexcept { /* TODO could be broken */
            using std::swap;
            swap(lhs.data, rhs.data);
            swap(lhs.batch_size, rhs.batch_size);
            swap(lhs.num_chans, rhs.num_chans);
            swap(lhs.height, rhs.height);
            swap(lhs.width, rhs.width);
        }

    private:
        /* TODO stop this mutable abuse */
        mutable common_data<value_type> data;
        mutable size_type batch_size, num_chans, width, height;
    };

    template <class T> inline
    bool have_same_shape(const tensor<T>& lhs, const tensor<T>& rhs) {
        return lhs.get_num_samples() == rhs.get_num_samples() &&
            lhs.get_chans() == rhs.get_chans() &&
            lhs.get_height() == rhs.get_height() &&
            lhs.get_width() == rhs.get_width();
    }

    template <class T> inline
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

    template <class T> inline
    void add(cublas_context& handle, const tensor<T>& lhs, const tensor<T>& rhs, tensor<T>& result) {
        assert(have_same_shape(lhs, rhs));
        
        const auto dest_nr = result.get_height();
        const auto dest_nc = result.get_width();
        const auto lhs_nr = lhs.get_height();
        const auto rhs_nr = rhs.get_height();
        const auto rhs_nc = rhs.get_width();

        blas::geam<T>(handle, false, false, dest_nr, dest_nc, 1.0,
            lhs.get_device_readonly(), lhs_nr, 1.0, rhs.get_device_readonly(), rhs_nr,
            result.get_device_writeable(), dest_nr);
    }

    template <class T> inline
    void abs(const tensor<T>& src, tensor<T>& dest) {
        dest.resize(src.get_num_samples(), src.get_chans(), src.get_height(), src.get_width());
        math::abs(view<T>(src.get_device_readonly(), src.size()),
            span<T>(dest.get_device_writeable(), dest.size()));
    }

    template <class T> inline
    void bnll(const tensor<T>& src, tensor<T>& dest) {
        dest.resize(src.get_num_samples(), src.get_chans(), src.get_height(), src.get_width());
        math::bnll(view<T>(src.get_device_readonly(), src.size()),
            span<T>(dest.get_device_writeable(), dest.size()));
    }

    template <class T> inline
    void relu(const tensor<T>& src, tensor<T>& dest, T slope = 0) {
        dest.resize(src.get_num_samples(), src.get_chans(), src.get_height(), src.get_width());
        math::relu(view<T>(src.get_device_readonly(), src.size()),
            span<T>(dest.get_device_writeable(), dest.size()), slope);
    }
    
    template <class T> inline
    void clipped_relu(const tensor<T>& src, tensor<T>& dest, T max = 6, T min = 0) {
        dest.resize(src.get_num_samples(), src.get_chans(), src.get_height(), src.get_width());
        math::clipped_relu(view<T>(src.get_device_readonly(), src.size()),
            span<T>(dest.get_device_writeable(), dest.size()), max, min);
    }

    template <class T> inline
    void channelwise_relu(const tensor<T>& src, tensor<T>& dest, const tensor<T>& slope) {
        dest.resize(src.get_num_samples(), src.get_chans(), src.get_height(), src.get_width());
        auto inner_size = src.get_width() * src.get_height();
        for (std::size_t c = 0; c < src.get_chans(); c++) {
            auto src_ptr = src.get_device_readonly() + c * inner_size;
            auto dest_ptr = dest.get_device_writeable() + c * inner_size;
            math::relu(view<T>(src_ptr, inner_size), span<T>(dest_ptr, inner_size), slope.read(c));
        }        
    }

    template <class T> inline
    void elu(const tensor<T>& src, tensor<T>& dest, T alpha) {
        dest.resize(src.get_num_samples(), src.get_chans(), src.get_height(), src.get_width());
        math::elu(view<T>(src.get_device_readonly(), src.size()),
            span<T>(dest.get_device_writeable(), dest.size()), alpha);
    }

    template <class T> inline
    void power(const tensor<T>& src, tensor<T>& dest, T exp = 1, T scale = 1, T shift = 0) {
        dest.resize(src.get_num_samples(), src.get_chans(), src.get_height(), src.get_width());
        math::power(view<T>(src.get_device_readonly(), src.size()),
            span<T>(dest.get_device_writeable(), dest.size()), exp, scale, shift);
    }

    template <class T> inline
    void sigmoid(const tensor<T>& src, tensor<T>& dest) {
        dest.resize(src.get_num_samples(), src.get_chans(), src.get_height(), src.get_width());
        math::sigmoid(view<T>(src.get_device_readonly(), src.size()),
            span<T>(dest.get_device_writeable(), dest.size()));
    }

    template <class T> inline
    void tanh(const tensor<T>& src, tensor<T>& dest) {
        dest.resize(src.get_num_samples(), src.get_chans(), src.get_height(), src.get_width());
        math::tanh(view<T>(src.get_device_readonly(), src.size()),
            span<T>(dest.get_device_writeable(), dest.size()));
    }

    template <class T>
    class convolution : noncopyable {
    public:
        struct params_type {
            params_type() = default;
            params_type(std::size_t num, std::size_t channels, std::size_t kheight, std::size_t kwidth) noexcept
                : padding{ 0, 0 }, stride{ 1, 1 }, dialation{ 1, 1 },
                kernel{ num, channels, kheight, kwidth } { }

            struct { std::size_t x, y; } padding, stride, dialation;
            struct {
                std::size_t num;
                std::size_t channels;
                std::size_t height;
                std::size_t width;
            } kernel;
        };

        convolution() = default;
        convolution(convolution&&) = default;
        convolution(std::size_t N, std::size_t in_height, std::size_t in_width, params_type& params) {
            auto& kparams = params.kernel;

            inputTensorDesc = cudnn::tensor_descriptor<T>(N, kparams.channels, in_height, in_width);

            filterDesc = cudnn::filter_descriptor<T>(kparams.num, kparams.channels, kparams.height, kparams.width);

            auto& padding = params.padding;
            auto& stride = params.stride;
            auto& dialation = params.dialation;
            convDesc = cudnn::convolution_descriptor<T>(padding.y, padding.x, stride.y, stride.x, dialation.y, dialation.x);

            get_convolution_output_dim(convDesc, filterDesc, inputTensorDesc,
                                       output_n, output_chans, output_height, output_width);

            biasTensorDesc = cudnn::tensor_descriptor<T>(1, output_chans, 1, 1);
            outputTensorDesc = cudnn::tensor_descriptor<T>(output_n, output_chans, output_height, output_width);

            algo = cudnn::convolution_algorithm<T>(handle, convDesc, filterDesc, inputTensorDesc, outputTensorDesc);
        }

        convolution& operator=(convolution&&) = default;

        void set_workspace(workspace ws) {
            scratchpad = std::move(ws);
            scratchpad.require(algo.get_workspace_size());
        }

        void convolve(const tensor<T>& input, const tensor<T>& kernels, tensor<T>& output) {
            assert(scratchpad.size() >= algo.get_workspace_size());
            output.resize(output_n, output_chans, output_height, output_width);
            cudnn::convolve<T>(handle, filterDesc, kernels.get_device_readonly(), convDesc, algo, scratchpad.get(),
               inputTensorDesc, input.get_device_readonly(), 1.0, 0.0, outputTensorDesc, output.get_device_writeable());
        }

        void convolve(const tensor<T>& input, const tensor<T>& kernels, const tensor<T>& bias, tensor<T>& output) {
            convolve(input, kernels, output);
            cudnn::add_tensor(handle, biasTensorDesc, bias.get_device_readonly(), outputTensorDesc, output.get_device_writeable());
        }

    private:
        cudnn::handle handle;

        cudnn::tensor_descriptor<T> inputTensorDesc, outputTensorDesc;
        cudnn::tensor_descriptor<T> biasTensorDesc;

        cudnn::filter_descriptor<T> filterDesc;
        cudnn::convolution_descriptor<T> convDesc;

        cudnn::convolution_algorithm<T> algo;

        std::size_t output_n, output_chans, output_height, output_width;

        workspace scratchpad;
    };

    template <class T>
    class pooling : noncopyable {
    public:
        struct params_type {
            std::size_t window_height, window_width;
            struct { std::size_t x, y; } padding, stride; 
        };

        pooling() = default;
        pooling(pooling&&) = default;
        pooling(cudnn::pooling_type type, std::size_t N, std::size_t in_chans, std::size_t in_height, std::size_t in_width, params_type& params) {

            inputTensorDesc = cudnn::tensor_descriptor<T>(N, in_chans, in_height, in_width);

            poolingDesc = cudnn::pooling_descriptor(type, params.window_height, params.window_width,
                                                       params.padding.x, params.padding.y,
                                                       params.stride.x, params.stride.y);

            get_pooling_output_dim(poolingDesc, inputTensorDesc, output_n, output_chans, output_height, output_width);

            outputTensorDesc = cudnn::tensor_descriptor<T>(output_n, output_chans, output_height, output_width);
        }

        pooling& operator=(pooling&&) = default;

        void pool(const tensor<T>& input, tensor<T>& output) {
            output.resize(output_n, output_chans, output_height, output_width);
            cudnn::pool<T>(handle, poolingDesc, inputTensorDesc, input.get_device_readonly(), 1.0, 0.0, outputTensorDesc, output.get_device_writeable());
        }

    private:
        cudnn::handle handle;

        cudnn::tensor_descriptor<T> inputTensorDesc, outputTensorDesc;
        cudnn::pooling_descriptor poolingDesc;
 
        std::size_t output_n, output_chans, output_height, output_width;
    };

    template <class T> inline
    void softmax(const tensor<T>& input, tensor<T>& output, bool log = false) {
        output.resize(input.get_num_samples(), input.get_chans(), input.get_height(), input.get_width());

        cudnn::handle handle;

        using cudnn::tensor_descriptor;
        tensor_descriptor<T> input_desc(input.get_num_samples(), input.get_chans(), input.get_height(), input.get_width());
        tensor_descriptor<T> output_desc(output.get_num_samples(), output.get_chans(), output.get_height(), output.get_width());
        cudnn::softmax(handle, input_desc, input.get_device_readonly(), output_desc, output.get_device_writeable(), log);
    }

    template <class T> inline
    void matrix_to_tensor(const matrix<T>& src, tensor<T>& dest) {
        dest.resize(src.get_n(), src.get_chans(), src.get_height(), src.get_width());
        for (int n = 0; n < src.get_n(); n++)
            for (int c = 0; c < src.get_chans(); c++)
                for (int i = 0; i < src.get_height(); i++)
                    for (int j = 0; j < src.get_width(); j++)
                        dest.write(n, c, i, j, src.at(n, c, i, j));
    }

    template <class T> inline
    void tensor_to_matrix(const tensor<T>& src, matrix<T>& dest) {
        dest.resize(src.get_num_samples(), src.get_chans(), src.get_height(), src.get_width());
        for (int n = 0; n < src.get_num_samples(); n++)
            for (int c = 0; c < src.get_chans(); c++)
                for (int i = 0; i < src.get_height(); i++)
                    for (int j = 0; j < src.get_width(); j++)
                        dest.at(n, c, i, j) = src.read(n, c, i, j);
    }
}

#endif /* CUDA_TENSOR_HPP */