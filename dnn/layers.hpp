#ifndef DNN_LAYERS_HPP
#define DNN_LAYERS_HPP

#include "matrix.hpp"
#include "cuda/tensor.hpp"

#include <memory>
#include <string>
#include <map>

namespace dnn {
    enum class layer_type {
        fully_connected,
        softmax
    };

    template <class T>
    class layer_params {
    public:
        std::map<std::string, matrix<T>> matrix;
        std::map<std::string, int> values;
    };

    template <class T>
    class layer {
    public:
        virtual ~layer() { }
        virtual void set_params(const layer_params<T>& params) { };
        virtual void forward(const cuda::tensor<T>& input, cuda::tensor<T>& output) = 0;
    };

    template <class T>
    class fully_connected : public layer<T> {
    public:
        fully_connected() noexcept
            : num_inputs{ 0 }, num_outputs{ 0 }, has_bias{ false }  { };
        fully_connected(cuda::stream strm) noexcept
            : stream(std::move(strm)), num_inputs{ 0 }, num_outputs{ 0 }, has_bias{ false }  { };

        void set_params(const layer_params<T>& params) override {
            assert(params.values.count("num_inputs") > 0);
            assert(params.values.count("num_outputs") > 0);
            assert(params.values.count("has_bias") > 0);
            assert(params.matrix.count("weights") > 0);

            num_outputs = params.values.at("num_outputs");
            num_inputs = params.values.at("num_inputs");
            has_bias = params.values.at("has_bias");

            const auto& weights_source = params.matrix.at("weights");
            weights.resize(1, 1, num_inputs, num_outputs);
            for (int i = 0; i < num_inputs; i++)
                for (int j = 0; j < num_outputs; j++)
                    weights.write(i, j, weights_source.at(i, j));

            if (has_bias) {
                bias.resize(1, 1, 1, num_outputs);
                const auto& bias_source = params.matrix.at("bias");
                for (int i = 0; i < num_outputs; i++)
                    bias.write(i, bias_source.at(i));
            }
        }

        void forward(const cuda::tensor<T>& input, cuda::tensor<T>& output) override {
            assert(num_inputs > 0 && num_outputs > 0); /* TODO exception */

            output.resize(1, 1, 1, num_outputs);

            cuda::cublas_context handle(stream);
            cuda::multiply(handle, weights, input, output);
            if (has_bias)
                cuda::add(handle, output, bias, output);
        }

    private:
        bool has_bias;
        std::size_t num_inputs, num_outputs;
        cuda::tensor<T> weights, bias;
        cuda::stream stream;
    };

    template <class T>
    class softmax : public layer<T> {
    public:
        softmax() = default;
        
        void set_params(const layer_params<T>& params) override { }

        void forward(const cuda::tensor<T>& input, cuda::tensor<T>& output) override {
            output.resize(1, 1, input.get_width(), input.get_height());
            cuda::softmax(input, output);
        }
    };
}

#endif /* DNN_LAYERS_HPP */