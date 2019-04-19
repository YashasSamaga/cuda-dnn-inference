#ifndef CUDA_CUBLAS_HPP
#define CUDA_CUBLAS_HPP

#include <cublas_v2.h>

#include "error.hpp"
#include "../utils/noncopyable.hpp"

namespace cuda {
    class cublas_context : noncopyable {
    public:
        cublas_context() { CHECK_CUBLAS(cublasCreate(&handle)); }
        ~cublas_context() { CHECK_CUBLAS(cublasDestroy(handle)); }

        auto get() const noexcept { return handle; }

    protected:
        cublasHandle_t handle;
    };
}

#endif /* CUDA_CUBLAS_HPP */