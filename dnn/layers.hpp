#ifndef DNN_LAYERS_HPP
#define DNN_LAYERS_HPP

#include "matrix.hpp"
#include "cuda/tensor.hpp"
#include "cuda/cudnn.hpp"
#include "cuda/workspace.hpp"

#include <memory>
#include <string>
#include <map>

namespace dnn {
    enum class layer_type {
        fully_connected,
        softmax,
        convolution,

        /* activation layers */
        abs,
        bnll,
        elu,
        power,
        relu,
        sigmoid,
        tanh
    };

    template <class T>
    class layer_params {
    public:
        std::map<std::string, matrix<T>> matrix;
        std::map<std::string, T> values;
        std::map<std::string, int> integers;
    };

    template <class T>
    class layer {
    public:
        virtual ~layer() { }
        virtual void set_params(const layer_params<T>& params) { };
        virtual void forward(const cuda::tensor<T>& input, cuda::tensor<T>& output, cuda::workspace& scratchpad) = 0;
    };

    template <class T>
    class fully_connected_layer : public layer<T> {
    public:
        fully_connected_layer() noexcept
            : num_inputs{ 0 }, num_outputs{ 0 }, has_bias{ false }  { };

        void set_params(const layer_params<T>& params) override {
            assert(params.matrix.count("weights") > 0);
            has_bias = params.matrix.count("bias");

            /* NOTE we infer input/output shape from the weight matrix
            ** `weights` shape = (1, 1, h, w)
            ** `bias` shape = (1, 1, h, w)
            ** `input` is reshaped to a column vector (n, 1, h, 1)
            ** `output` = `weights` x `input` + `bias`
            ** we reshape `output` to (n, h, 1, 1)
            */
            const auto& wsrc = params.matrix.at("weights");
            assert(wsrc.get_n() == 1 && wsrc.get_chans() == 1); /* or reshape? */
            cuda::matrix_to_tensor(wsrc, weights);

            if (has_bias) {
                const auto& bsrc = params.matrix.at("bias");
                assert(bsrc.size() == bsrc.get_height());
                cuda::matrix_to_tensor(bsrc, bias);
            }
        }

        void forward(const cuda::tensor<T>& input, cuda::tensor<T>& output, cuda::workspace& scratchpad) override {
            assert(input.get_num_samples() == 1); /* batch processing not supported yet TODO */

            const auto num_input = weights.get_width(),
                       num_output = weights.get_height();

            auto innerSize = input.size() / input.get_num_samples();
            assert(innerSize == num_input);

            input.reshape(input.get_num_samples(), 1, innerSize, 1);

            cuda::cublas_context handle(stream); /* get rid of this TODO */

            output.resize(input.get_num_samples(), 1, num_output, 1);
            for (std::size_t n = 0; n < input.get_num_samples(); n++) {
                /* TODO need to use tensor_view/span and process one sample at a time */
                cuda::multiply(handle, weights, input, output);
                if (has_bias)
                    cuda::add(handle, output, bias, output);
            }
            output.reshape(input.get_num_samples(), num_output, 1, 1);
        }

    private:
        bool has_bias;
        std::size_t num_inputs, num_outputs;
        cuda::tensor<T> weights, bias;
        cuda::stream stream;
    };

    template <class T>
    class softmax_layer : public layer<T> {
    public:
        softmax_layer() : log{ false } { };
        
        void set_params(const layer_params<T>& params) override {
            if (params.integers.count("log"))
                log = params.integers.at("log");
        }

        void forward(const cuda::tensor<T>& input, cuda::tensor<T>& output, cuda::workspace& scratchpad) override {
            /* softmax performed channnel wise */
            /* input shape = output_shape */
            cuda::softmax(input, output, log);
        }

    private:
        bool log; /* true => compute log probabilities, false => compute probabilities */
    };

    template <class T>
    class abs_layer : public layer<T> {
    public:
        void forward(const cuda::tensor<T>& input, cuda::tensor<T>& output, cuda::workspace& scratchpad) override {
            cuda::abs(input, output);
        }
    };

    template <class T>
    class bnll_layer : public layer<T> {
    public:
        void forward(const cuda::tensor<T>& input, cuda::tensor<T>& output, cuda::workspace& scratchpad) override {
            cuda::bnll(input, output);
        }
    };

    template <class T>
    class elu_layer : public layer<T> {
    public:
        elu_layer() : alpha{ 1 } { }

        void set_params(const layer_params<T>& params) override {
            if (params.values.count("alpha"))
                alpha = params.values.at("alpha");
        }

        void forward(const cuda::tensor<T>& input, cuda::tensor<T>& output, cuda::workspace& scratchpad) override {
            cuda::elu(input, output, alpha);
        }

    private:
        T alpha;
    };

    template <class T>
    class power_layer : public layer<T> {
    public:
        power_layer() : exp{ 1 }, scale{ 1 }, shift{ 0 } { }

        void set_params(const layer_params<T>& params) override {
            if (params.values.count("exp"))
                exp = params.values.at("exp");
            if (params.values.count("scale"))
                scale = params.values.at("scale");
            if (params.values.count("shift"))
                shift = params.values.at("shift");
        }

        void forward(const cuda::tensor<T>& input, cuda::tensor<T>& output, cuda::workspace& scratchpad) override {
            cuda::power(input, output, exp, scale, shift);
        }

    private:
        T exp, scale, shift;
    };

    template <class T>
    class relu_layer : public layer<T> {
    public:
        relu_layer() : slope{ 0 } { }

        void set_params(const layer_params<T>& params) override {
            if (params.values.count("slope"))
                slope = params.values.at("slope");
        }

        void forward(const cuda::tensor<T>& input, cuda::tensor<T>& output, cuda::workspace& scratchpad) override {
            cuda::relu(input, output, slope);
        }

    private:
        T slope;
    };

    template <class T>
    class sigmoid_layer : public layer<T> {
    public:
        void forward(const cuda::tensor<T>& input, cuda::tensor<T>& output, cuda::workspace& scratchpad) override {
            cuda::sigmoid(input, output);
        }
    };

    template <class T>
    class tanh_layer : public layer<T> {
    public:
        void forward(const cuda::tensor<T>& input, cuda::tensor<T>& output, cuda::workspace& scratchpad) override {
            cuda::tanh(input, output);
        }
    };

    template <class T>
    class convolution_layer : public layer<T> {
    public:
        convolution_layer() = default;

        void set_params(const layer_params<T>& params) override {
            assert(params.matrix.count("weights") > 0);
            has_bias = params.matrix.count("bias");            

            const auto& wsrc = params.matrix.at("weights"); /* OC x IC x Kh x Kw */
            cuda::matrix_to_tensor(wsrc, filters);

            if (has_bias) {
                const auto& bsrc = params.matrix.at("bias"); /* 1 x OC x 1 x 1 */
                cuda::matrix_to_tensor(bsrc, bias);
            }

            const auto out_channels = filters.get_num_samples(); /* number of kernels */
            const auto in_channels = filters.get_chans();
            const auto kernel_height = filters.get_height();
            const auto kernel_width = filters.get_width();

            std::size_t padding_x = 0, padding_y = 0;
            if (params.integers.count("padding_x"))
                padding_x = params.integers.at("padding_x");
            if (params.integers.count("padding_y"))
                padding_y = params.integers.at("padding_y");

            std::size_t stride_x = 1, stride_y = 1;
            if (params.integers.count("stride_x"))
                stride_x = params.integers.at("stride_x");
            if (params.integers.count("stride_y"))
                stride_y = params.integers.at("stride_y");

            std::size_t dialation_x = 1, dialation_y = 1;
            if (params.integers.count("dialation_x"))
                dialation_x = params.integers.at("dialation_x");
            if (params.integers.count("dialation_y"))
                dialation_y = params.integers.at("dialation_y");

            conparams = cuda::convolution<T>::params_type(out_channels, in_channels, kernel_height, kernel_width);
            conparams.padding.x = padding_x;
            conparams.padding.y = padding_y;
            conparams.stride.x = stride_x;
            conparams.stride.y = stride_y;
            conparams.dialation.x = dialation_x;
            conparams.dialation.y = dialation_y;

            /*
            const auto kernel_extent_width = dialation_x * (kernel_width - 1) + 1;
            const auto kernel_extent_height = dialation_y * (kernel_height - 1) + 1;

            const auto out_width = (input_width + 2 * padding_x - kernel_extent_width) / stride_x + 1;
            const auto out_height = (input_height + 2 * padding_y - kernel_extent_height) / stride_y + 1;
            */
        }           

        void forward(const cuda::tensor<T>& input, cuda::tensor<T>& output, cuda::workspace& scratchpad) override {
            const auto in_channels = input.get_chans(),
                       in_height = input.get_height(),
                       in_width = input.get_width();
            assert(in_channels == conparams.kernel.channels);

            convoluter = cuda::convolution<T>(input.get_num_samples(), in_height, in_width, conparams);
            convoluter.set_workspace(scratchpad);
            if (has_bias)
                convoluter.convolve(input, filters, bias, output);
            else
                convoluter.convolve(input, filters, output);
        }

    private:
        bool has_bias;
   
        cuda::tensor<T> filters, bias;

        typename cuda::convolution<T>::params_type conparams;
        cuda::convolution<T> convoluter;
    };
}

#endif /* DNN_LAYERS_HPP */