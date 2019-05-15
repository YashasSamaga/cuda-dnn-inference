#ifndef DNN_NETWORK_HPP
#define DNN_NETWORK_HPP

#include <iostream>

#include "layers.hpp"
#include "cuda/workspace.hpp"

#include "utils/make_unique.hpp"

#include <vector>
#include <memory>

namespace dnn {
    template <class T = float>
    class network {
    public:
        void add_layer(layer_type type, layer_params<T>& params) {
            auto create_layer = [this](layer_type type)->std::unique_ptr<layer<T>> {
                switch (type) {
                case layer_type::fully_connected:
                    return make_unique<fully_connected_layer<T>>();
                case layer_type::softmax:
                    return make_unique<softmax_layer<T>>();
                case layer_type::convolution:
                    return make_unique<convolution_layer<T>>();
                case layer_type::abs:
                    return make_unique<abs_layer<T>>();
                case layer_type::bnll:
                    return make_unique<bnll_layer<T>>();
                case layer_type::elu:
                    return make_unique<elu_layer<T>>();
                case layer_type::power:
                    return make_unique<power_layer<T>>();
                case layer_type::relu:
                    return make_unique<relu_layer<T>>();
                case layer_type::clipped_relu:
                    return make_unique<clipped_relu_layer<T>>();
                case layer_type::channelwise_relu:
                    return  make_unique<channelwise_relu_layer<T>>();
                case layer_type::sigmoid:
                    return make_unique<sigmoid_layer<T>>();
                case layer_type::tanh:
                    return make_unique<tanh_layer<T>>();
                }
                return nullptr;
            };

            auto ptr = create_layer(type);
            ptr->set_params(params);

            layers.push_back(std::move(ptr));
        }

        void add_layer(layer_type type) {
            layer_params<T> params;
            add_layer(type, params);
        }

        void forward(const matrix<T>& input, matrix<T>& output) {
            cuda::tensor<T> input_tensor, output_tensor;
            cuda::matrix_to_tensor(input, input_tensor);

            for (auto& ptr : layers) {
                ptr->forward(input_tensor, output_tensor, scratchpad);
                input_tensor = std::move(output_tensor);
            }
            output_tensor = std::move(input_tensor);
            
            cuda::tensor_to_matrix(output_tensor, output);
            stream.synchronize(); /* broken FIX */
        }

    private:
        cuda::stream stream;
        cuda::workspace scratchpad;
        std::vector<std::unique_ptr<layer<T>>> layers; /* TODO use graph */
    };
}
#endif /* DNN_NETWORK_HPP */