#ifndef CUDA_ERROR_HPP
#define CUDA_ERROR_HPP

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <sstream>
#include <exception>
#include <string>
#include <cassert>

#define CHECK_CUDA(call) cuda::detail::check_cuda_status((call), __FILE__, __LINE__)  
#define CHECK_CUBLAS(call) cuda::detail::check_cublas_status((call), __FILE__, __LINE__) 

namespace cuda {
    namespace detail {
        class exception : public std::exception {
        public:
            explicit exception(const char* msg) : what_msg(msg) { }
            explicit exception(const std::string& msg) : what_msg(msg) { }
            virtual ~exception() { }

            const char* what() const noexcept override { return what_msg.c_str(); }

        protected:
            std::string what_msg;
        };
    }
    
    class cuda_exception : public detail::exception {
    public:
        using detail::exception::exception;
    };

    class cublas_exception : public detail::exception {
    public:
        using detail::exception::exception;
    };

    namespace detail {
        void check_cuda_status(cudaError_t error, std::string filename, std::size_t lineno) {
            if (error != cudaSuccess) {                                                
                std::ostringstream stream;                                             
                stream << "CUDA Error: " << __FILE__ << ":" << __LINE__ << '\n';       
                stream << cudaGetErrorString(error) << '\n';                           
                throw cuda::cuda_exception(stream.str());                                   
            }
        }
    
        void check_cublas_status(cublasStatus_t error, std::string filename, std::size_t lineno) {
            auto cublasGetErrorString = [](cublasStatus_t err) {
                switch (err) {
                    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
                    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
                    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
                    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
                    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
                    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
                    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
                    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
                }
                return "UNKNOWN_CUBLAS_ERROR";
            };

            if (error != CUBLAS_STATUS_SUCCESS) {                                                
                std::ostringstream stream;                                             
                stream << "CUBLAS Error: " << __FILE__ << ":" << __LINE__ << '\n';       
                stream << cublasGetErrorString(error) << '\n';                           
                throw cuda::cublas_exception(stream.str());                                   
            }
        }
    }
}

#endif /* CUDA_ERROR_HPP */