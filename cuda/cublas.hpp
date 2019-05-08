#ifndef CUDA_CUBLAS_HPP
#define CUDA_CUBLAS_HPP

#include "error.hpp"
#include "stream.hpp"
#include "pointer.hpp"
#include "utils/noncopyable.hpp"

#include <cublas_v2.h>

#include <sstream>
#include <string>
#include <cstddef>
#include <cassert>

#define CHECK_CUBLAS(call) cuda::detail::check_cublas_status((call), __FILE__, __LINE__) 

namespace cuda {
    class cublas_exception : public exception {
    public:
        using exception::exception;
    };

    namespace detail {
        inline void check_cublas_status(cublasStatus_t error, std::string filename, std::size_t lineno) {
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
                stream << "CUBLAS Error: " << filename << ":" << lineno << '\n';
                stream << cublasGetErrorString(error) << '\n';
                throw cuda::cublas_exception(stream.str());
            }
        }
    }

    class cublas_context : noncopyable {
    public:
        cublas_context() { CHECK_CUBLAS(cublasCreate(&handle)); }
        cublas_context(stream s) : strm(std::move(s)) { 
            CHECK_CUBLAS(cublasCreate(&handle));
            CHECK_CUBLAS(cublasSetStream(handle, strm.get()));
        }
        ~cublas_context() { CHECK_CUBLAS(cublasDestroy(handle)); }

        auto get() const noexcept { return handle; }

    private:
        cublasHandle_t handle;
        stream strm;
    };

    namespace blas {
        template <class T>
        void gemm(cublas_context& handle, bool transa, bool transb, std::size_t rows_a, std::size_t cols_b, std::size_t cols_a,
            T alpha, const device_ptr<const T> A, std::size_t lda, const device_ptr<const T> B, std::size_t ldb,
            T beta, const device_ptr<T> C, std::size_t ldc);

        template <> inline
        void gemm<float>(cublas_context& handle, bool transa, bool transb, std::size_t rows_a, std::size_t cols_b, std::size_t cols_a,
           float alpha, const device_ptr<const float> A, std::size_t lda, const device_ptr<const float> B, std::size_t ldb,
           float beta, const device_ptr<float> C, std::size_t ldc) 
        {
            auto opa = transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                opb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
            int irows_a = static_cast<int>(rows_a),
                icols_b = static_cast<int>(cols_b),
                icols_a = static_cast<int>(cols_a),
                ilda = static_cast<int>(lda),
                ildb = static_cast<int>(ldb),
                ildc = static_cast<int>(ldc);
            if (!transa && !transb) {
                CHECK_CUBLAS(cublasSgemm(handle.get(), opa, opb, icols_b, irows_a, icols_a, &alpha, B.get(), ildb, A.get(), ilda, &beta, C.get(), ildc));
                return;
            }

            assert(0); /* unsupported */
        }

        template <> inline
        void gemm<double>(cublas_context& handle, bool transa, bool transb, std::size_t rows_a, std::size_t cols_b, std::size_t cols_a,
            double alpha, const device_ptr<const double> A, std::size_t lda, const device_ptr<const double> B, std::size_t ldb,
            double beta, const device_ptr<double> C, std::size_t ldc)
        {
            auto opa = transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                opb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
            int irows_a = static_cast<int>(rows_a),
                icols_b = static_cast<int>(cols_b),
                icols_a = static_cast<int>(cols_a),
                ilda = static_cast<int>(lda),
                ildb = static_cast<int>(ldb),
                ildc = static_cast<int>(ldc);
            if (!transa && !transb) {
                CHECK_CUBLAS(cublasDgemm(handle.get(), opa, opb, irows_a, icols_b, icols_a, &alpha, A.get(), ilda, B.get(), ildb, &beta, C.get(), ildc));
                return;
            }

            assert(0); /* unsupported */
        }

        template <class T>
        void geam(cublas_context& handle, bool transa, bool transb, std::size_t rows_c, std::size_t cols_c, T alpha,
            const device_ptr<const T> A, std::size_t lda, T beta, const device_ptr<const T> B, std::size_t ldb,
            const device_ptr<T> C, std::size_t ldc);

        template <> inline
        void geam<float>(cublas_context& handle, bool transa, bool transb, std::size_t rows, std::size_t cols, float alpha,
            const device_ptr<const float> A, std::size_t lda, float beta, const device_ptr<const float> B, std::size_t ldb,
            const device_ptr<float> C, std::size_t ldc) 
        {
            auto opa = transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                opb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
            int irows = static_cast<int>(rows),
                icols = static_cast<int>(cols),
                ilda = static_cast<int>(lda),
                ildb = static_cast<int>(ldb),
                ildc = static_cast<int>(ldc);
            CHECK_CUBLAS(cublasSgeam(handle.get(), opa, opb, irows, icols, &alpha, A.get(), ilda, &beta, B.get(), ildb, C.get(), ildc));
        }

        template <> inline
        void geam<double>(cublas_context& handle, bool transa, bool transb, std::size_t rows, std::size_t cols, double alpha,
            const device_ptr<const double> A, std::size_t lda, double beta, const device_ptr<const double> B, std::size_t ldb,
            const device_ptr<double> C, std::size_t ldc) 
        {
            auto opa = transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                opb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
            int irows = static_cast<int>(rows),
                icols = static_cast<int>(cols),
                ilda = static_cast<int>(lda),
                ildb = static_cast<int>(ldb),
                ildc = static_cast<int>(ldc);
            CHECK_CUBLAS(cublasDgeam(handle.get(), opa, opb, irows, icols, &alpha, A.get(), ilda, &beta, B.get(), ildb, C.get(), ildc));
        }
    }
}

#endif /* CUDA_CUBLAS_HPP */