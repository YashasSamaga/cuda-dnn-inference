#ifndef CUDA_CUDNN_HPP
#define CUDA_CUDNN_HPP

#include <cuda_runtime.h>
#include <cudnn.h>

#include "error.hpp"
#include "../utils/noncopyable.hpp"

namespace cuda {
    class cudnn_context : noncopyable {
    public:
        cudnn_context() { CHECK_CUDNN(cudnnCreate(&handle)); }
        ~cudnn_context() { CHECK_CUDNN(cudnnDestroy(handle)); }

        auto get() const noexcept { return handle; }

    protected:
        cudnnHandle_t handle;
    };
}

#endif /* CUDA_CUDNN_HPP */