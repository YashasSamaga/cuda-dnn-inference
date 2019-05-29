#ifndef CUDA_CUDNN_HPP
#define CUDA_CUDNN_HPP

#include "error.hpp"
#include "stream.hpp"
#include "utils/noncopyable.hpp"

#include <cudnn.h>

#include <sstream>
#include <string>
#include <cstddef>

#define CHECK_CUDNN(call) cuda::cudnn::detail::check_cudnn_status((call), __FILE__, __LINE__)

namespace cuda {
    namespace cudnn {
        /* what a mess; do something about these names TODO */
        class exception : public cuda::exception {
        public:
            using cuda::exception::exception;
        };

        namespace detail {
            inline void check_cudnn_status(cudnnStatus_t error, std::string filename, std::size_t lineno) {
                if (error != CUDNN_STATUS_SUCCESS) {
                    std::ostringstream stream;
                    stream << "CUDNN Error: " << filename << ":" << lineno << '\n';
                    stream << cudnnGetErrorString(error) << '\n';
                    throw cuda::cudnn::exception(stream.str());
                }
            }
        }

        class handle : noncopyable {
        public:
            handle() { CHECK_CUDNN(cudnnCreate(&hndl)); }  
            handle(handle&&) = default;
            handle(stream s) : strm(std::move(s)) {
                CHECK_CUDNN(cudnnCreate(&hndl));
                try {
                    CHECK_CUDNN(cudnnSetStream(hndl, strm.get()));
                }
                catch (...) {
                    CHECK_CUDNN(cudnnDestroy(hndl));
                    throw;
                }
            }
            ~handle() { if(hndl != nullptr) CHECK_CUDNN(cudnnDestroy(hndl)); }

            handle& operator=(handle&& other) {
                hndl = other.hndl;
                other.hndl = nullptr;
                return *this;
            }

            auto get() const noexcept { return hndl; }

        private:
            cudnnHandle_t hndl;
            stream strm;
        };

        enum class data_format {
            nchw,
            nhwc
        };

        namespace detail {
            /* get_data_type<T> returns the equivalent cudnn enumeration constant
            ** for example, get_data_type<T> returns CUDNN_DATA_FLOAT
            */
            template <class> auto get_data_type()->decltype(CUDNN_DATA_FLOAT);
            template <> inline auto get_data_type<float>()->decltype(CUDNN_DATA_FLOAT) { return CUDNN_DATA_FLOAT; }
            template <> inline auto get_data_type<double>()->decltype(CUDNN_DATA_FLOAT) { return CUDNN_DATA_DOUBLE; }

            /* get_data_format<T> returns the equivalent cudnn enumeration constant
            ** for example, get_data_type<data_format::nchw> returns CUDNN_TENSOR_NCHW
            */
            template <data_format> auto get_data_format()->decltype(CUDNN_TENSOR_NCHW);
            template <> inline auto get_data_format<data_format::nchw>()->decltype(CUDNN_TENSOR_NCHW) { return CUDNN_TENSOR_NCHW; }
            template <> inline auto get_data_format<data_format::nhwc>()->decltype(CUDNN_TENSOR_NCHW) { return CUDNN_TENSOR_NHWC; }
        }

        /* RAII wrapper for cudnnTensorDescriptor_t */
        template <class T, data_format format = data_format::nchw>
        class tensor_descriptor : noncopyable {
        public:
            tensor_descriptor() noexcept : descriptor{ nullptr } { }
            tensor_descriptor(tensor_descriptor&& other)
                : descriptor{ other.descriptor } {
                other.descriptor = nullptr;
            }

            tensor_descriptor(std::size_t N, std::size_t chans, std::size_t height, std::size_t width) {
                CHECK_CUDNN(cudnnCreateTensorDescriptor(&descriptor));
                try {
                    CHECK_CUDNN(cudnnSetTensor4dDescriptor(descriptor,
                        detail::get_data_format<format>(), detail::get_data_type<T>(),
                        static_cast<int>(N), static_cast<int>(chans),
                        static_cast<int>(height), static_cast<int>(width)));
                } catch (...) {
                    CHECK_CUDNN(cudnnDestroyTensorDescriptor(descriptor));
                    throw;
                }
            }

            ~tensor_descriptor() { /* destructor throws */
                if (descriptor != nullptr) {
                    CHECK_CUDNN(cudnnDestroyTensorDescriptor(descriptor));
                }
            }

            tensor_descriptor& operator=(tensor_descriptor&& other) noexcept {
                descriptor = other.descriptor;
                other.descriptor = nullptr;
                return *this;
            };

            auto get() const noexcept { return descriptor; }

        private:
            cudnnTensorDescriptor_t descriptor;
        };

        /* RAII wrapper for cudnnFilterDescriptor_t */
        template <class T, data_format format = data_format::nchw>
        class filter_descriptor : noncopyable {
        public:
            filter_descriptor() noexcept : descriptor{ nullptr } { }
            filter_descriptor(filter_descriptor&& other)
                : descriptor{ other.descriptor } {
                other.descriptor = nullptr;
            }

            filter_descriptor(std::size_t output_chans, std::size_t input_chans, std::size_t height, std::size_t width) {
                CHECK_CUDNN(cudnnCreateFilterDescriptor(&descriptor));
                try {
                    CHECK_CUDNN(cudnnSetFilter4dDescriptor(descriptor,
                        detail::get_data_type<T>(), detail::get_data_format<format>(),
                        static_cast<int>(output_chans), static_cast<int>(input_chans),
                        static_cast<int>(height), static_cast<int>(width)));
                } catch (...) {
                    CHECK_CUDNN(cudnnDestroyFilterDescriptor(descriptor));
                    throw;
                }
            }

            ~filter_descriptor() { /* destructor throws */
                if (descriptor != nullptr) {
                    CHECK_CUDNN(cudnnDestroyFilterDescriptor(descriptor));
                }
            }

            filter_descriptor& operator=(filter_descriptor&& other) noexcept {
                descriptor = other.descriptor;
                other.descriptor = nullptr;
                return *this;
            };

            auto get() const noexcept { return descriptor; }

        private:
            cudnnFilterDescriptor_t descriptor;
        };

        /* RAII wrapper for cudnnConvolutionDescriptor_t */
        template <class T>
        class convolution_descriptor : noncopyable {
        public:
            convolution_descriptor() noexcept : descriptor{ nullptr } { }
            convolution_descriptor(convolution_descriptor&& other)
                : descriptor{ other.descriptor } {
                other.descriptor = nullptr;
            }

            convolution_descriptor(std::size_t padding_y, std::size_t padding_x,
                                   std::size_t stride_y, std::size_t stride_x,
                                   std::size_t dilation_y, std::size_t dialation_x,
                                   std::size_t groups) {
                CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&descriptor));
                try {
                    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(descriptor,
                        static_cast<int>(padding_y), static_cast<int>(padding_x),
                        static_cast<int>(stride_y), static_cast<int>(stride_x),
                        static_cast<int>(dilation_y), static_cast<int>(dialation_x),
                        CUDNN_CROSS_CORRELATION, detail::get_data_type<T>()));
                    CHECK_CUDNN(cudnnSetConvolutionGroupCount(descriptor, groups));
                } catch (...) {
                    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(descriptor));
                    throw;
                }
            }

            ~convolution_descriptor() { /* destructor throws */
                if (descriptor != nullptr) {
                    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(descriptor));
                }
            }

            convolution_descriptor& operator=(convolution_descriptor&& other) noexcept {
                descriptor = other.descriptor;
                other.descriptor = nullptr;
                return *this;
            };

            auto get() const noexcept { return descriptor; }

        private:
            cudnnConvolutionDescriptor_t descriptor;
        };

        template <class T>
        class convolution_algorithm {
        public:
            convolution_algorithm() : workspace_size{ 0 } { }
            convolution_algorithm(convolution_algorithm&) = default;
            convolution_algorithm(convolution_algorithm&&) = default;

            convolution_algorithm& operator=(convolution_algorithm&) = default;
            convolution_algorithm& operator=(convolution_algorithm&&) = default;

            template <data_format format = data_format::nchw>
            convolution_algorithm(cudnn::handle& handle,
                convolution_descriptor<T>& conv,
                filter_descriptor<T, format>& filter,
                tensor_descriptor<T, format>& input,
                tensor_descriptor<T, format>& output) {
                CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(handle.get(),
                    input.get(), filter.get(), conv.get(), output.get(),
                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                    0, &algo));

                CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle.get(),
                    input.get(), filter.get(), conv.get(), output.get(),
                    algo, &workspace_size));
            }

            auto get() const noexcept { return algo; }

            auto get_workspace_size() const noexcept { return workspace_size; }

        private:
            cudnnConvolutionFwdAlgo_t algo;
            std::size_t workspace_size;
        };

        enum class pooling_type {
            max,
            average_exclude_padding,
            average_include_padding
        };

        namespace detail {
            /* get_pooling_type<T> returns the equivalent cudnn enumeration constant */
            auto get_pooling_type(pooling_type type) {
                switch (type) {
                case pooling_type::max:
                    return CUDNN_POOLING_MAX;
                case pooling_type::average_exclude_padding:
                    return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
                case pooling_type::average_include_padding:
                    return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
                }
                return CUDNN_POOLING_MAX; /* TODO how else do I handle? throw? */
            }
        }

        /* RAII wrapper for cudnnPoolingDescriptor */
        class pooling_descriptor : noncopyable {
        public:
            pooling_descriptor() noexcept : descriptor{ nullptr } { }
            pooling_descriptor(pooling_descriptor&& other)
                : descriptor{ other.descriptor } {
                other.descriptor = nullptr;
            }

            pooling_descriptor(pooling_type type,
                               std::size_t window_height, std::size_t window_width,
                               std::size_t padding_y, std::size_t padding_x,
                               std::size_t stride_y, std::size_t stride_x) {
                CHECK_CUDNN(cudnnCreatePoolingDescriptor(&descriptor));
                try {
                    CHECK_CUDNN(cudnnSetPooling2dDescriptor(descriptor, detail::get_pooling_type(type), CUDNN_PROPAGATE_NAN,
                        static_cast<int>(window_height), static_cast<int>(window_width),
                        static_cast<int>(padding_y), static_cast<int>(padding_x),
                        static_cast<int>(stride_y), static_cast<int>(stride_x)));
                } catch (...) {
                    CHECK_CUDNN(cudnnDestroyPoolingDescriptor(descriptor));
                    throw;
                }
            }

            ~pooling_descriptor() { /* destructor throws */
                if (descriptor != nullptr) {
                    CHECK_CUDNN(cudnnDestroyPoolingDescriptor(descriptor));
                }
            }

            pooling_descriptor& operator=(pooling_descriptor&& other) noexcept {
                descriptor = other.descriptor;
                other.descriptor = nullptr;
                return *this;
            };

            auto get() const noexcept { return descriptor; }

        private:
            cudnnPoolingDescriptor_t descriptor;
        };

        template <class T, data_format format = data_format::nchw> inline
        void get_convolution_output_dim(convolution_descriptor<T>& conv,
                                        filter_descriptor<T, format>& filter,
                                        tensor_descriptor<T, format>& input,
                                        std::size_t& n, std::size_t& c, std::size_t& h, std::size_t& w) {
            int in, ic, ih, iw;
            CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv.get(), input.get(), filter.get(), &in, &ic, &ih, &iw));
            n = static_cast<std::size_t>(in);
            c = static_cast<std::size_t>(ic);
            h = static_cast<std::size_t>(ih);
            w = static_cast<std::size_t>(iw);
        }

        template <class T, data_format format = data_format::nchw> inline
        void get_pooling_output_dim(pooling_descriptor& pooling_desc,
                tensor_descriptor<T, format>& input,
                std::size_t& n, std::size_t& c, std::size_t& h, std::size_t& w) {
            int in, ic, ih, iw;
            CHECK_CUDNN(cudnnGetPooling2dForwardOutputDim(pooling_desc.get(), input.get(), &in, &ic, &ih, &iw));
            n = static_cast<std::size_t>(in);
            c = static_cast<std::size_t>(ic);
            h = static_cast<std::size_t>(ih);
            w = static_cast<std::size_t>(iw);
        }

        /* cuDNN requires alpha/beta to be in float for half precision
        ** hence, half precision is unsupported (due to laziness)
        ** (this is what std::enable_if is doing there)
        */
        template <class T, data_format format = data_format::nchw, class U = unsigned char> inline
            typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value, void>
            ::type pool(cudnn::handle& handle,
                pooling_descriptor& pooling_desc,
                tensor_descriptor<T, format>& input_desc,
                device_ptr<const T> input_data,
                T alpha, T beta,
                tensor_descriptor<T, format>& output_desc,
                device_ptr<T> output_data) {
            CHECK_CUDNN(cudnnPoolingForward(handle.get(), pooling_desc.get(), &alpha,
                input_desc.get(), input_data.get(), &beta, output_desc.get(), output_data.get()));
        }

        /* cuDNN requires alpha/beta to be in float for half precision
        ** hence, half precision is unsupported (due to laziness)
        ** (this is what std::enable_if is doing there)
        */
        template <class T, data_format format = data_format::nchw, class U = unsigned char> inline
            typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value, void>
            ::type convolve(cudnn::handle& handle,
                        filter_descriptor<T, format>& filter_desc,
                        device_ptr<const T> kernels,

                        convolution_descriptor<T>& conv_desc,
                        convolution_algorithm<T>& algo,
                        device_ptr<U> workspace,
                            
                        tensor_descriptor<T, format>& input_desc,
                        device_ptr<const T> input_data,
                        T alpha,
                        T beta,
                        tensor_descriptor<T, format>& output_desc,
                        device_ptr<T> output_data) {
            CHECK_CUDNN(cudnnConvolutionForward(handle.get(), &alpha, input_desc.get(), input_data.get(),
                filter_desc.get(), kernels.get(), conv_desc.get(), algo.get(), workspace.get(),
                algo.get_workspace_size(), &beta, output_desc.get(), output_data.get()));
        }

        /* cuDNN requires alpha/beta to be in float for half precision
        ** hence, half precision is unsupported (due to laziness)
        ** (this is what std::enable_if is doing there)
        */
        template <class T, data_format format = data_format::nchw> inline
            typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value, void>
            ::type add_tensor(cudnn::handle& handle,
                tensor_descriptor<T, format>& bias_desc, const device_ptr<const T> bias_data,
                tensor_descriptor<T, format>& output_desc, const device_ptr<T> output_data) {
            T alpha = 1.0, beta = 1.0;
            CHECK_CUDNN(cudnnAddTensor(handle.get(), &alpha, bias_desc.get(), bias_data.get(), &beta, output_desc.get(), output_data.get()));
        }

        /* cuDNN requires alpha/beta to be in float for half precision
        ** hence, half precision is unsupported (due to laziness)
        ** (this is what std::enable_if is doing there)
        */
        template <class T, data_format format = data_format::nchw> inline
            typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value, void>
            ::type softmax(cudnn::handle& handle, 
                           tensor_descriptor<T, format>& input_desc, device_ptr<const T> input_data,
                           tensor_descriptor<T, format>& output_desc, device_ptr<T> output_data,
                           bool log = false) {
            T alpha = 1.0, beta = 0.0;
            cudnnSoftmaxAlgorithm_t algo = log ? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE;
            CHECK_CUDNN(cudnnSoftmaxForward(handle.get(), algo, CUDNN_SOFTMAX_MODE_CHANNEL,
                &alpha, input_desc.get(), input_data.get(), &beta, output_desc.get(), output_data.get()));
        }
    }    
}

#endif /* CUDA_CUDNN_HPP */