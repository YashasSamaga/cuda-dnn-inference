#ifndef CUDA_ERROR_HPP
#define CUDA_ERROR_HPP

#include <cuda_runtime.h>

#include <sstream>
#include <exception>
#include <string>

#define ASSERT(expr) cuda::detail::check_assertion((expr), #expr, __FILE__, __LINE__) 
#define CHECK_CUDA(call) cuda::detail::check_cuda_status((call), __FILE__, __LINE__)  

namespace cuda {
    class exception : public std::exception {
    public:
        explicit exception(std::string msg) : what_msg(std::move(msg)) { }
        virtual ~exception() { }

        const char* what() const noexcept override { return what_msg.c_str(); }

    protected:
        std::string what_msg;
    };

    class cuda_exception : public exception {
    public:
        using exception::exception;
    };

    namespace detail {
        inline void check_cuda_status(cudaError_t error, std::string filename, std::size_t lineno) {
            if (error != cudaSuccess) {
                std::ostringstream stream;
                stream << "CUDA Error: " << filename << ":" << lineno << '\n';
                stream << cudaGetErrorString(error) << '\n';
                throw cuda::cuda_exception(stream.str());
            }
        }

        inline void check_assertion(bool cond, std::string expression, std::string filename, std::size_t lineno) {
            if (!cond) {
                std::ostringstream stream;
                stream << "Assertion Failure: " << expression << '\n';
                stream << "At line " << lineno << "in file " << filename << '\n';
                throw cuda::exception(stream.str());
            }
        }
    }
}

#endif /* CUDA_ERROR_HPP */