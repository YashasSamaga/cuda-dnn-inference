#include <cuda_runtime.h>

#include "span.hpp"
#include "utils.hpp"

namespace cuda {
    namespace math {
        template <class T>
        __global__ void exp(view<T> input, span<T> output) {
            assert(input.size() <= output.size());
            for (auto i : grid_stride_range(input.size())) {
                output[i] = ::exp(input[i]);
            }
        }

        template __global__ void exp(view<float>, span<float>);
        template __global__ void exp(view<double>, span<double>);

        template <class T>
        __global__ void softmax_warpwise(view<T> input, span<T> output, const device_ptr<T> temp) {
            assert(input.size() <= output.size());
            T sum = 0.0;
            for (auto i : grid_stride_range(input.size())) {
                output[i] = ::exp(input[i]);
                sum += output[i];
            }
            
            /* collect sum from all threads of the warp in lane zero */
            for (int mask = warpSize / 2; mask > 0; mask /= 2)
                sum += __shfl_down(sum, mask);

            /* add sum from all the warps */
            T* total = temp.get();
            if (threadIdx.x % warpSize == 0)
                atomicAdd(total, sum);

            /* take the ratio to obtain probabilities */
            for (auto i : grid_stride_range(input.size()))
                output[i] /= *total;
        }

        template __global__ void softmax_warpwise(view<float>, span<float>, const device_ptr<float>);
        // TODO template __global__ void softmax_warpwise(view<double>, span<double>, const device_ptr<double>);
    }
}

