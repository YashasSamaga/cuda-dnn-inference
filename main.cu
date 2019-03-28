#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cudnn.lib")

#include <cuda_runtime.h>

#include "cuda/utils.hpp"
#include "cuda/memory.hpp"
#include "cuda/cublas.hpp"

#include "benchmark.hpp"

#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <iomanip>

/* cpu implementations */
namespace cpu {
    template <class T>
    void vector_add(const T* first1, const T* first2, T* d_first, std::size_t N) {
        for (std::size_t i = 0; i < N; i++)
            d_first[i] = first1[i] + first2[i];
    }

    template <class T>
    void matrix_add(const T* first, const T* second, T* result, std::size_t nx, std::size_t ny) {
        for (std::size_t i = 0; i < ny; i++) {
            for (std::size_t j = 0; j < nx; j++) {
                const auto idx = i * nx + j;
                result[idx] = first[idx] + second[idx];
            }
        }
    }

    template <class T>
    void matrix_multiply(const T* first, const T* second, T* result, std::size_t n) {
        for (std::size_t i = 0; i < n; i++) {
            for (std::size_t j = 0; j < n; j++) {
                const auto idx = j * n + i;
                result[idx] = T(0);
                for (std::size_t k = 0; k < n; k++) {
                    const auto first_idx = i * n + k,
                               second_idx = k * n + j;
                    result[idx] += first[first_idx] * second[second_idx];
                }
            }
        }
    }
}

/* custom gpu implementations */
namespace gpu {
    template <class T>
    __global__ void vector_add(const T* first1, const T* first2, T* d_first, std::size_t N) {
        for(auto i : cuda::grid_stride_range(0, N))
            d_first[i] = first1[i] + first2[i];
    }

    template <class T>
    __global__ void matrix_add(const T* first, const T* second, T* result, std::size_t nx, std::size_t ny) {
        for (auto idx : cuda::grid_stride_range(0, nx * ny)) {
            result[idx] = first[idx] + second[idx];
        }
    }

    template <class T>
    __global__ void matrix_multiply(const T* first, const T* second, T* result, std::size_t n) {
        for (auto i : cuda::grid_stride_range_x(n)) {
            for (auto j : cuda::grid_stride_range_y(n)) {
                const auto idx = j * n + i;
                result[idx] = T(0); /* TODO CHECK PTX becaz using temporary variable slows down */
                for (std::size_t k = 0; k < n; k++) {
                    const auto first_idx = i * n + k,
                               second_idx = k * n + j;
                    result[idx] += first[first_idx] * second[second_idx];
                }
            }
        }        
    }
}

/* cublas implementation */
namespace cublas {
    template <class T>
    void matrix_multiply(cuda::cublas_handle& handle,  const T* first, const T* second, T* result, std::size_t n) {
        static_assert(std::is_same<T, float>::value, "uses cublasSgemm; hence, requires T to be float");

        int in = static_cast<int>(n);
        const float alpha = 1.0, beta = 0.0;
        cublasSgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_T,
                    in, in, in,
                    &alpha,
                    first, in,
                    second, in,
                    &beta,
                    result, in);
        cublasSgeam(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, in, in, &alpha, result, in, &beta, nullptr, in, result, in);        
    }

    template <class T>
    void matrix_add(cuda::cublas_handle& handle, const T* first, const T* second, T* result, std::size_t nx, std::size_t ny) {
        static_assert(std::is_same<T, float>::value, "uses cublasSgeam; hence, requires T to be float");

        int inx = static_cast<int>(nx), iny = static_cast<int>(ny);
        const float alpha = 1.0, beta = 1.0;
        cublasSgeam(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                    iny, inx,
                    &alpha,
                    first, inx,
                    &beta,
                    second, inx,
                    result, inx);
    }
}

template <class T>
auto to_milliseconds(const T& duration) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration);
}

template <class T, class ForwardItr>
auto check_result(ForwardItr first1, ForwardItr last1, ForwardItr first2, T ratio) {
    return std::mismatch(first1, last1, first2, [ratio](auto lhs, auto rhs) {
        return std::fabs(rhs - lhs) / std::min(rhs, lhs) < ratio;
    });
}

template <class Container>
void random_fill(Container& cont) {
    static std::random_device rd;
    static std::mt19937 rng(rd());
    static std::uniform_real_distribution<typename Container::value_type> dist(1.0, 1000.0);
    std::generate(std::begin(cont), std::end(cont), []() { return dist(rng); });
}

void test_matrix_multiply() {
    using T = float;

    constexpr std::size_t n = 1 << 12, size = n * n;

    /* generate sample data */
    std::vector<T> lhs(size), rhs(size);
    random_fill(lhs);
    random_fill(rhs);

    /* run on cpu */
    std::vector<T> cpu_result(size);
    auto cpu_time = benchmark([&lhs, &rhs, &cpu_result] () {
        //cpu::matrix_multiply(lhs.data(), rhs.data(), &cpu_result[0], n);
    });    
    std::cout << "CPU Time: " << to_milliseconds(cpu_time).count() << "ms" << std::endl;

    /* setup GPU */
    cuda::managed_ptr<T> d_lhs, d_rhs, d_result;
    auto gpu_prep_time = benchmark([&d_lhs, &d_rhs, &d_result, &lhs, &rhs]() {
        d_lhs.reset(size);
        d_rhs.reset(size);
        d_result.reset(size);
        cuda::memcpy(d_lhs, lhs.data(), size * sizeof(T));
        cuda::memcpy(d_rhs, rhs.data(), size * sizeof(T));
    });
    std::cout << "GPU Preparation Time: " << to_milliseconds(gpu_prep_time).count() << "ms" << std::endl;

    /* run custom GPU kernel */
    auto gpu_time = benchmark([&d_lhs, &d_rhs, &d_result]() {
        dim3 block(32, 32);
        dim3 grid((n + block.x - 1)/block.x, (n + block.y - 1)/block.y);
        cuda::launch_kernel(gpu::matrix_multiply<T>, grid, block, d_lhs.get(), d_rhs.get(), d_result.get(), n);
        /* cuda::launch_kernel(gpu::matrix_multiply<T>, d_lhs.get(), d_rhs.get(), d_result.get(), n); */
        cuda::device_synchronize();
    });

    std::vector<T> gpu_result(size);
    cuda::memcpy(&gpu_result[0], d_result, size * sizeof(T));
    std::cout << "GPU Time: " << to_milliseconds(gpu_time).count() << "ms" << std::endl;    
    
    auto pr = check_result(std::begin(cpu_result), std::end(cpu_result), std::begin(gpu_result), 0.001);
    bool match = (pr.first == std::end(cpu_result));
    std::cout << "CPU and GPU output " << (match ? "match" : "do not match") << std::endl;
    if (!match) {
        std::cout << std::setprecision(std::numeric_limits<T>::digits10 + 1);
        std::cout << "Mismatch: "<< *pr.first << " " << *pr.second << std::endl;
    }

    cuda::memset(d_result, 0, size * sizeof(T));
    cuda::device_synchronize();

    cuda::cublas_handle handle; /* declared outside because lazy initialization screws with the benchmarks */
    auto cublas_time = benchmark([&handle, &d_lhs, &d_rhs, &d_result]() {
        cublas::matrix_multiply(handle, d_lhs.get(), d_rhs.get(), d_result.get(), n);
        cuda::device_synchronize();
    });
    cuda::memcpy(&gpu_result[0], d_result, size * sizeof(T));
    std::cout << "CUBLAS Time: " << to_milliseconds(cublas_time).count() << "ms" << std::endl;    
    
    pr = check_result(std::begin(cpu_result), std::end(cpu_result), std::begin(gpu_result), T(0.02));
    match = (pr.first == std::end(cpu_result));
    std::cout << "CPU and CUBLAS output " << (match ? "match" : "do not match") << std::endl;
    if (!match) {
        std::cout << std::setprecision(std::numeric_limits<T>::digits10 + 1);
        std::cout << "Mismatch: "<< *pr.first << " " << *pr.second << std::endl;
    }  
}

void test_matrix_add() {
    using T = float;

    constexpr int n = 1 << 14, size = n * n;

    std::vector<T> lhs(size), rhs(size);
    random_fill(lhs);
    random_fill(rhs);

    std::vector<T> cpu_result(size);
    auto cpu_time = benchmark([&lhs, &rhs, &cpu_result] () {
        cpu::matrix_add(lhs.data(), rhs.data(), &cpu_result[0], n, n);
    });    
    std::cout << "CPU Time: " << to_milliseconds(cpu_time).count() << "ms" << std::endl;

    cuda::managed_ptr<T> d_lhs, d_rhs, d_result;
    auto gpu_prep_time = benchmark([&d_lhs, &d_rhs, &d_result, &lhs, &rhs]() {
        d_lhs.reset(size);
        d_rhs.reset(size);
        d_result.reset(size);
        cuda::memcpy(d_lhs, lhs.data(), size * sizeof(T));
        cuda::memcpy(d_rhs, rhs.data(), size * sizeof(T));
    });
    std::cout << "GPU Preparation Time: " << to_milliseconds(gpu_prep_time).count() << "ms" << std::endl;

    auto gpu_time = benchmark([&d_lhs, &d_rhs, &d_result]() {
        cuda::launch_kernel(gpu::matrix_add<T>, d_lhs.get(), d_rhs.get(), d_result.get(), n, n);
        cuda::device_synchronize();
    });
    std::vector<T> gpu_result(size);
    cuda::memcpy(&gpu_result[0], d_result, size * sizeof(T));
    std::cout << "GPU Time: " << to_milliseconds(gpu_time).count() << "ms" << std::endl;    
    
    auto pr = check_result(std::begin(cpu_result), std::end(cpu_result), std::begin(gpu_result), T(0.02));
    bool match = pr.first == std::end(cpu_result);
    std::cout << "CPU and GPU output " << (match ? "match" : "do not match") << std::endl;
    if (!match) {
        std::cout << std::setprecision(std::numeric_limits<T>::digits10 + 1);
        std::cout << "Mismatch: "<< *pr.first << " " << *pr.second << std::endl;
    }

    cuda::memset(d_result, 0, size * sizeof(T));
    cuda::device_synchronize();

    cuda::cublas_handle handle; /* declared outside because lazy initialization screws with the benchmarks */
    auto cublas_time = benchmark([&handle, &d_lhs,&d_rhs, &d_result]() {
        cublas::matrix_add(handle, d_lhs.get(), d_rhs.get(), d_result.get(), n, n);
        cuda::device_synchronize();
    });
    cuda::memcpy(&gpu_result[0], d_result, size * sizeof(T));
    std::cout << "CUBLAS Time: " << to_milliseconds(cublas_time).count() << "ms" << std::endl;    
    
    pr = check_result(std::begin(cpu_result), std::end(cpu_result), std::begin(gpu_result), T(0.02));
    match = (pr.first == std::end(cpu_result));
    std::cout << "CPU and CUBLAS output " << (match ? "match" : "do not match") << std::endl;
    if (!match) {
        std::cout << std::setprecision(std::numeric_limits<T>::digits10 + 1);
        std::cout << "Mismatch: "<< *pr.first << " " << *pr.second << std::endl;
    }
}

void test_vector_add() {
    using T = float;

    constexpr int N = 1 << 26;

    std::vector<T> lhs(N), rhs(N);
    random_fill(lhs);
    random_fill(rhs);

    std::vector<T> cpu_result(N);
    auto cpu_time = benchmark([&lhs, &rhs, &cpu_result] () {
        cpu::vector_add(lhs.data(), rhs.data(), &cpu_result[0], N);
    });    
    std::cout << "CPU Time: " << to_milliseconds(cpu_time).count() << "ms" << std::endl;

    cuda::managed_ptr<T> d_lhs, d_rhs, d_result;
    auto gpu_prep_time = benchmark([&d_lhs, &d_rhs, &d_result, &lhs, &rhs]() {
        d_lhs.reset(N);
        d_rhs.reset(N);
        d_result.reset(N);
        cuda::memcpy(d_lhs, lhs.data(), N * sizeof(T));
        cuda::memcpy(d_rhs, rhs.data(), N * sizeof(T));
    });
    std::cout << "GPU Preparation Time: " << to_milliseconds(gpu_prep_time).count() << "ms" << std::endl;

    auto gpu_time = benchmark([&d_lhs,&d_rhs, &d_result]() {
        cuda::launch_kernel(gpu::vector_add<T>, d_lhs.get(), d_rhs.get(), d_result.get(), N);
        cuda::device_synchronize();
    });
    std::vector<T> gpu_result(N);
    cuda::memcpy(&gpu_result[0], d_result, N * sizeof(T));
    std::cout << "GPU Time: " << to_milliseconds(gpu_time).count() << "ms" << std::endl;    
    
    auto pr = check_result(std::begin(cpu_result), std::end(cpu_result), std::begin(gpu_result), T(0.02));
    bool match = (pr.first == std::end(cpu_result));
    std::cout << "CPU and GPU output " << (match ? "match" : "do not match") << std::endl;
    if (!match) {
        std::cout << std::setprecision(std::numeric_limits<T>::digits10 + 1);
        std::cout << "Mismatch: "<< *pr.first << " " << *pr.second << std::endl;
    }
}

int main() {
    int dev = 0;
    cudaDeviceProp properties;
    CHECK_CUDA(cudaGetDeviceProperties(&properties, dev));
    std::cout << "Device: " << dev << ", " << properties.name << '\n' << std::endl;

    CHECK_CUDA(cudaSetDevice(dev));
    CHECK_CUDA(cudaFree(0)); /* establish context beforehand so that the benchmarks are not disturbed */

    std::cout << "Vector Addition:\n";
    test_vector_add();
    std::cout << std::endl;

    std::cout << "Matrix Addition:\n";
    test_matrix_add();
    std::cout << std::endl;

    std::cout << "Matrix Multiplication:\n";
    test_matrix_multiply();

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}