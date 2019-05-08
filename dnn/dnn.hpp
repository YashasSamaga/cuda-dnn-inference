#ifndef DNN_HPP
#define DNN_HPP

#include "matrix.hpp"
#include "cuda/tensor.hpp"

#include "utils/make_unique.hpp"

#include <vector>
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

    template <class T = float>
    class network {
    public:
        void add_layer(layer_type type, layer_params<T>& params) {
            auto create_layer = [this](layer_type type)->std::unique_ptr<layer<T>> {
                switch (type) {
                case layer_type::fully_connected:
                    return make_unique<fully_connected<T>>(stream);
                }
                return nullptr;
            };

            auto ptr = create_layer(type);
            ptr->set_params(params);
            layers.push_back(std::move(ptr));
        }

        void forward(const matrix<T>& input, matrix<T>& output) {
            cuda::tensor<T> input_tensor, output_tensor;
            input_tensor.resize(1, 1, input.get_cols(), input.get_rows());

            /* TODO eliminate redundant copy 
            ** matrix -> tensor_host -> tensor_gpu 
            ** to
            ** matrix -> tensor_gpu
            */
            for (int i = 0; i < input.get_rows(); i++)
                for (int j = 0; j < input.get_cols(); j++)
                    input_tensor.write(i, j, input.at(i, j)); 

            for (auto& ptr : layers) {
                ptr->forward(input_tensor, output_tensor);
                input_tensor = std::move(output_tensor);
            }

            output_tensor = std::move(input_tensor);
            output.resize(output_tensor.get_height(), output_tensor.get_width());
            for (int i = 0; i < output_tensor.get_width(); i++)
                for (int j = 0; j < output_tensor.get_height(); j++)
                    output.at(i, j) = output_tensor.read(i, j);
            stream.synchronize();
        }

    private:
        cuda::stream stream;
        std::vector<std::unique_ptr<layer<T>>> layers; /* TODO use graph */
    };
}

#endif /* DNN_HPP */