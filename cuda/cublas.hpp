#ifndef CUDA_CUBLAS_HPP
#define CUDA_CUBLAS_HPP

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "error.hpp"

namespace cuda {
    class cublas_handle {
    public:
        cublas_handle() { CHECK_CUBLAS(cublasCreate(&handle)); }
        ~cublas_handle() { CHECK_CUBLAS(cublasDestroy(handle)); }
        cublas_handle(const cublas_handle&) = delete;
        cublas_handle& operator=(const cublas_handle&) = delete;

        auto get() const noexcept { return handle; }

    protected:
        cublasHandle_t handle;
    };
}

#endif /* CUDA_CUBLAS_HPP */