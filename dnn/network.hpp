#ifndef DNN_NETWORK_HPP
#define DNN_NETWORK_HPP

#include <iostream>

#include "layers.hpp"

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
                    return make_unique<fully_connected<T>>(stream);
                case layer_type::softmax:
                    return make_unique<softmax<T>>();
                }
                return nullptr;
            };

            auto ptr = create_layer(type);
            ptr->set_params(params);
            layers.push_back(std::move(ptr));
        }

        void forward(const matrix<T>& input, matrix<T>& output) {
            cuda::tensor<T> input_tensor(stream), output_tensor(stream);
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
#endif /* DNN_NETWORK_HPP */