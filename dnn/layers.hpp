#ifndef DNN_LAYERS_HPP
#define DNN_LAYERS_HPP
#include <iostream>
#include "matrix.hpp"
#include "cuda/tensor.hpp"

#include <memory>
#include <string>
#include <map>

namespace dnn {
    enum class layer_type {
        fully_connected,
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
        virtual void set_params(layer_params<T> params) { };
        virtual void forward(const cuda::tensor<T>& input, cuda::tensor<T>& output) = 0;
    };

    template <class T>
    class fully_connected : public layer<T> {
    public:
        fully_connected() noexcept
            : num_inputs{ 0 }, num_outputs{ 0 }, has_bias{ false }  { };
        fully_connected(cuda::stream strm) noexcept
            : stream(std::move(strm)), num_inputs{ 0 }, num_outputs{ 0 }, has_bias{ false }  { };

        void set_params(layer_params<T> params) override {
            assert(params.values.count("num_inputs") > 0);
            assert(params.values.count("num_outputs") > 0);
            assert(params.values.count("has_bias") > 0);
            assert(params.matrix.count("weights") > 0);

            num_outputs = params.values["num_outputs"];
            num_inputs = params.values["num_inputs"];
            has_bias = params.values["has_bias"];

            weights.resize(1, 1, num_inputs, num_outputs);
            for (int i = 0; i < num_inputs; i++)
                for (int j = 0; j < num_outputs; j++)
                    weights.write(i, j, params.matrix["weights"].at(i, j));

            if (has_bias) {
                bias.resize(1, 1, 1, num_outputs);
                for (int i = 0; i < num_outputs; i++)
                    bias.write(i, params.matrix["bias"].at(i));
            }
        }

        void forward(const cuda::tensor<T>& input, cuda::tensor<T>& output) override {
            assert(num_inputs > 0 && num_outputs > 0); /* TODO exception */

            output.resize(1, 1, 1, num_outputs);

            cuda::cublas_context handle(strm);
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
}

#endif /* DNN_LAYERS_HPP */