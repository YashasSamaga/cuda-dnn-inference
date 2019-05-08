#ifndef CUDA_CUDNN_HPP
#define CUDA_CUDNN_HPP

#include "error.hpp"
#include "stream.hpp"
#include "utils/noncopyable.hpp"

#include <cudnn.h>

#include <sstream>
#include <string>
#include <cstddef>

#define CHECK_CUDNN(call) cuda::detail::check_cudnn_status((call), __FILE__, __LINE__)

namespace cuda {
    class cudnn_exception : public exception {
    public:
        using exception::exception;
    };

    namespace detail {
        inline void check_cudnn_status(cudnnStatus_t error, std::string filename, std::size_t lineno) {
            if (error != cudaSuccess) {
                std::ostringstream stream;
                stream << "CUDA Error: " << filename << ":" << lineno << '\n';
                stream << cudnnGetErrorString(error) << '\n';
                throw cuda::cudnn_exception(stream.str());
            }
        }
    }

    class cudnn_context : noncopyable {
    public:
        cudnn_context() { CHECK_CUDNN(cudnnCreate(&handle)); }
        cudnn_context(stream s) : strm(std::move(s)) { 
            CHECK_CUDNN(cudnnCreate(&handle));
            CHECK_CUDNN(cudnnSetStream(handle, strm.get()));
        }

        ~cudnn_context() { CHECK_CUDNN(cudnnDestroy(handle)); }

        auto get() const noexcept { return handle; }

    private:
        cudnnHandle_t handle;
        stream strm;
    };
}

#endif /* CUDA_CUDNN_HPP */