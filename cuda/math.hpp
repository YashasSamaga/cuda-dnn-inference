#include "span.hpp"

namespace cuda {
    namespace math {
        template <class T>
        __global__ void exp(view<T> input, span<T> output);

        template <class T>
        __global__ void softmax_warpwise(view<T> input, span<T> output, const device_ptr<T> temp);
    }
}