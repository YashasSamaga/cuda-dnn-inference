#ifndef CUDA_ERROR_HPP
#define CUDA_ERROR_HPP

#include <cuda_runtime.h>
#include <sstream>
#include <exception>
#include <string>

#define CHECK_CUDA(call)                                                       \
do {                                                                           \
    const auto error = call;                                                   \
    if (error != cudaSuccess) {                                                \
        std::ostringstream stream;                                             \
        stream << "CUDA Error: " << __FILE__ << ":" << __LINE__ << '\n';       \
        stream << cudaGetErrorString(error) << '\n';                            \
        throw cuda::exception(stream.str());                                   \
    }                                                                          \
} while (0);              

namespace cuda {
    class exception : public std::exception {
    public:
        explicit exception(const char* msg) : what_msg(msg) { }
        explicit exception(const std::string& msg) : what_msg(msg) { }
        virtual ~exception() { }

        virtual const char* what() const { return what_msg.c_str(); }

    protected:
        std::string what_msg;
    };
}

#endif /* CUDA_ERROR_HPP */