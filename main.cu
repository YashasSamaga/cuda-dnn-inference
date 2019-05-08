#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cudnn.lib")

#include "matrix.hpp"
#include "dnn/dnn.hpp"

#include "cuda/memory.hpp"
#include "cuda/data.hpp"
#include "cuda/cublas.hpp"
#include "cuda/utils.hpp"

#include "benchmark.hpp"

#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#if defined(__GNUC__) && ((__GNUC__ > 3) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#   define RESTRICT __restrict
#elif defined(_MSC_VER) && _MSC_VER >= 1400
#   define RESTRICT __restrict
#else
#   define RESTRICT
#endif

#define CUDA_RESTRICT __restrict__

/* cpu implementations */
namespace cpu {
    template <class T>
    void vector_add(const T* first, const T* second, T* result, std::size_t n) {
        for (std::size_t i = 0; i < n; i++)
            result[i] = first[i] + second[i];
    }

    template <class T>
    void matrix_add(const T* first, const T* second, T* result, std::size_t n) {
        for (std::size_t i = 0; i < n; i++) {
            for (std::size_t j = 0; j < n; j++) {
                const auto idx = i * n + j;
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
    __global__ void vector_add(cuda::view<T> first1, cuda::view<T> first2, cuda::span<T> d_first) {
        assert(first1.size() >= d_first.size());
        assert(first2.size() >= d_first.size());

		const auto n = d_first.size();
        for(auto i : cuda::grid_stride_range(n))
            d_first[i] = first1[i] + first2[i];
    }

    template <class T>
    __global__ void matrix_add(cuda::device_ptr<const T> first, cuda::device_ptr<const T> second, cuda::device_ptr<T> result, std::size_t n) {
        for (auto i : cuda::grid_stride_range(n * n))
            result[i] = first[i] + second[i];
    }

    template <class T>
    __global__ void matrix_multiply(cuda::device_ptr<const T> first, cuda::device_ptr<const T> second, cuda::device_ptr<T> result, std::size_t n) {
        for (auto i : cuda::grid_stride_range_x(n)) {
            for (auto j : cuda::grid_stride_range_y(n)) {
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

/* cublas implementation */
namespace cublas {
    template <class T>
    void matrix_add(cuda::cublas_context& handle, cuda::device_ptr<const T> first, cuda::device_ptr<const T> second, cuda::device_ptr<T> result, std::size_t n) {
        cuda::blas::geam<T>(handle, false, false, n, n, 1.0, first, n, 1.0, second, n, result, n);
    }

    template <class T>
    void matrix_multiply(cuda::cublas_context& handle, cuda::device_ptr<const T> first, cuda::device_ptr<const T> second, cuda::device_ptr<T> result, std::size_t n) {
        cuda::blas::gemm<T>(handle, false, false, n, n, n, 1.0f, first, n, second, n, 0.0, result, n);
    }
}

template <class T>
auto to_milliseconds(const T& duration) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration);
}

/* finds mismatches in two ranges
** a mismatch is obtained when the two corresponding values go beyond the specified relative error
*/
template <class T, class ForwardItr1, class ForwardItr2>
auto check_result(ForwardItr1 first1, ForwardItr1 last1, ForwardItr2 first2, T ratio) {
    return std::mismatch(first1, last1, first2, [ratio](auto lhs, auto rhs) {
        return std::fabs(rhs - lhs) / std::min(rhs, lhs) < ratio;
    });
}

template <class InputIt>
void random_fill(InputIt first, InputIt last) {
    using value_type = typename std::iterator_traits<InputIt>::value_type;

    static std::random_device rd;
	static std::mt19937 rng{ rd() };
    static std::uniform_real_distribution<value_type> dist(1.0, 1000.0);
    std::generate(first, last, [] { return dist(rng); });
}

void test_matrix_multiply() {
    using T = float;
    using allocator = cuda::pinned_allocator<T>;

    constexpr std::size_t n = 1 << 11, size = n * n;

    /* generate sample data */
    std::vector<T, allocator> lhs(size), rhs(size);
    random_fill(std::begin(lhs), std::end(lhs));
    random_fill(std::begin(rhs), std::end(rhs));

    /* run on cpu */
    std::vector<T, allocator> cpu_result(size);
    auto cpu_time = benchmark([&lhs, &rhs, &cpu_result] {
		/* too slow to test */
        //cpu::matrix_multiply(lhs.data(), rhs.data(), &cpu_result[0], n);
    });    
    std::cout << "[CPU] Computation Time: " << to_milliseconds(cpu_time).count() << "ms" << std::endl;

    /* setup GPU */
    cuda::managed_ptr<T> d_lhs, d_rhs, d_result;
    auto gpu_alloc_time = benchmark([&d_lhs, &d_rhs, &d_result] {
        d_lhs.reset(size);
        d_rhs.reset(size);
        d_result.reset(size);
    });
    std::cout << "[GPU] Allocation Time: " << to_milliseconds(gpu_alloc_time).count() << "ms" << std::endl;

    auto gpu_copy_time = benchmark([&d_lhs, &d_rhs, &lhs, &rhs] {
        cuda::memcpy(d_lhs, lhs.data());
        cuda::memcpy(d_rhs, rhs.data());
    });
    std::cout << "[GPU] Copy Time: " << to_milliseconds(gpu_copy_time).count() << "ms" << std::endl;

    /* run custom GPU kernel */
    auto gpu_comp_time = benchmark([&d_lhs, &d_rhs, &d_result] {
        auto block = dim3(32, 32);
        auto grid = dim3((n + block.x - 1)/block.x, (n + block.y - 1)/block.y);
        cuda::launch_kernel(gpu::matrix_multiply<T>, grid, block, d_lhs.get(), d_rhs.get(), d_result.get(), n);
        //cuda::launch_kernel(gpu::matrix_multiply<T>, d_lhs.get(), d_rhs.get(), d_result.get(), n);
        cuda::device_synchronize();
    });

    std::vector<T, allocator> gpu_result(size);
    cuda::memcpy(&gpu_result[0], d_result);
    std::cout << "[GPU] Computation Time: " << to_milliseconds(gpu_comp_time).count() << "ms" << std::endl;
    
    auto pr = check_result(std::begin(cpu_result), std::end(cpu_result), std::begin(gpu_result), 0.001);
    auto match = (pr.first == std::end(cpu_result));
    std::cout << "CPU and GPU output " << (match ? "match" : "do not match") << std::endl;
    if (!match) {
        std::cout << std::setprecision(std::numeric_limits<T>::digits10 + 1);
        std::cout << "Mismatch: "<< *pr.first << " " << *pr.second << std::endl;
    }

    cuda::memset(d_result, 0);
    cuda::device_synchronize();

    cuda::cublas_context handle; /* declared outside because lazy initialization screws with the benchmarks */
    auto cublas_time = benchmark([&handle, &d_lhs, &d_rhs, &d_result] {
        cublas::matrix_multiply<float>(handle, d_lhs.get(), d_rhs.get(), d_result.get(), n);
        cuda::device_synchronize();
    });
    cuda::memcpy(&gpu_result[0], d_result);
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
    using allocator = cuda::pinned_allocator<T>;

    constexpr int n = 1 << 14, size = n * n;

    std::vector<T, allocator> lhs(size), rhs(size);
    random_fill(std::begin(lhs), std::end(lhs));
    random_fill(std::begin(rhs), std::end(rhs));

    std::vector<T, allocator> cpu_result(size);
    auto cpu_time = benchmark([&lhs, &rhs, &cpu_result] {
        cpu::matrix_add(lhs.data(), rhs.data(), &cpu_result[0], n);
    });    
    std::cout << "[CPU] Computation Time: " << to_milliseconds(cpu_time).count() << "ms" << std::endl;

    cuda::managed_ptr<T> d_lhs, d_rhs, d_result;
    auto gpu_alloc_time = benchmark([&d_lhs, &d_rhs, &d_result] {
        d_lhs.reset(size);
        d_rhs.reset(size);
        d_result.reset(size);
    });
    std::cout << "[GPU] Allocation Time: " << to_milliseconds(gpu_alloc_time).count() << "ms" << std::endl;

    auto gpu_copy_time = benchmark([&d_lhs, &d_rhs, &lhs, &rhs] {
        cuda::memcpy(d_lhs, lhs.data());
        cuda::memcpy(d_rhs, rhs.data());
    });
    std::cout << "[GPU] Copy Time: " << to_milliseconds(gpu_copy_time).count() << "ms" << std::endl;

    auto gpu_comp_time = benchmark([&d_lhs, &d_rhs, &d_result] {
        cuda::launch_kernel(gpu::matrix_add<T>, d_lhs.get(), d_rhs.get(), d_result.get(), n);
        cuda::device_synchronize();
    });
    std::vector<T, allocator> gpu_result(size);
    cuda::memcpy(&gpu_result[0], d_result);
    std::cout << "[GPU] Computation Time: " << to_milliseconds(gpu_comp_time).count() << "ms" << std::endl;
    
    auto pr = check_result(std::begin(cpu_result), std::end(cpu_result), std::begin(gpu_result), T(0.02));
    auto match = (pr.first == std::end(cpu_result));
    std::cout << "CPU and GPU output " << (match ? "match" : "do not match") << std::endl;
    if (!match) {
        std::cout << std::setprecision(std::numeric_limits<T>::digits10 + 1);
        std::cout << "Mismatch: "<< *pr.first << " " << *pr.second << std::endl;
    }

    cuda::memset(d_result, 0);
    cuda::device_synchronize();

    cuda::cublas_context handle; /* declared outside because lazy initialization screws with the benchmarks */
    auto cublas_time = benchmark([&handle, &d_lhs,&d_rhs, &d_result] {
        cublas::matrix_add<T>(handle, d_lhs.get(), d_rhs.get(), d_result.get(), n);
        cuda::device_synchronize();
    });
    cuda::memcpy(&gpu_result[0], d_result);
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
    using allocator = cuda::pinned_allocator<T>;

    constexpr int N = 1 << 26;

    std::vector<T, allocator> lhs(N), rhs(N);
    random_fill(std::begin(lhs), std::end(lhs));
    random_fill(std::begin(rhs), std::end(rhs));

    std::vector<T, allocator> cpu_result(N);
    auto cpu_time = benchmark([&lhs, &rhs, &cpu_result] {
        cpu::vector_add(lhs.data(), rhs.data(), &cpu_result[0], N);
    });    
    std::cout << "[CPU] Computation Time: " << to_milliseconds(cpu_time).count() << "ms" << std::endl;

    cuda::managed_ptr<T> d_lhs, d_rhs, d_result;
    auto gpu_alloc_time = benchmark([&d_lhs, &d_rhs, &d_result, &lhs, &rhs] {
        d_lhs.reset(N);
        d_rhs.reset(N);
        d_result.reset(N);
    });
    std::cout << "[GPU] Allocation Time: " << to_milliseconds(gpu_alloc_time).count() << "ms" << std::endl;
    
    auto gpu_copy_time = benchmark([&d_lhs, &d_rhs, &d_result, &lhs, &rhs] {
        cuda::memcpy(d_lhs, lhs.data());
        cuda::memcpy(d_rhs, rhs.data());
    });
    std::cout << "[GPU] Copy Time: " << to_milliseconds(gpu_copy_time).count() << "ms" << std::endl;

    auto gpu_comp_time = benchmark([&d_lhs,&d_rhs, &d_result] {
        cuda::launch_kernel(gpu::vector_add<T>, d_lhs, d_rhs, d_result);
        cuda::device_synchronize();
    });
    std::vector<T, allocator> gpu_result(N);
    cuda::memcpy(&gpu_result[0], d_result);
    std::cout << "[GPU] Computation Time: " << to_milliseconds(gpu_comp_time).count() << "ms" << std::endl;
    
    auto pr = check_result(std::begin(cpu_result), std::end(cpu_result), std::begin(gpu_result), T(0.02));
    auto match = (pr.first == std::end(cpu_result));
    std::cout << "CPU and GPU output " << (match ? "match" : "do not match") << std::endl;
    if (!match) {
        std::cout << std::setprecision(std::numeric_limits<T>::digits10 + 1);
        std::cout << "Mismatch: "<< *pr.first << " " << *pr.second << std::endl;
    }
}

void test_data_transfer() {
    using T = float;

    /* testing fill & copy one after another might affect the results of the later 
    ** hence, we restrict the tests to either fill or copy at a time
    */
    bool test_fill = false;

    constexpr int size = 1 << 29;
    std::cout << "sample size: " << (size * sizeof(T)) / 1024 / 1024  << "MB\n\n";

    std::cout << "pageable memory:\n";
    {
        std::vector<T> sample;
        auto allocation_time = benchmark([&sample] {
            sample.resize(size);
        });
        std::cout << "allocation time = " << to_milliseconds(allocation_time).count() << "ms\n";

        switch (test_fill) {
            case true: {
                auto fill_time = benchmark([&sample] {
                    std::fill(std::begin(sample), std::end(sample), T(100.0));
                });
                std::cout << "fill time = " << to_milliseconds(fill_time).count() << "ms\n";
                break;
            }
            case false: {
                cuda::managed_ptr<T> dest(size);
                auto copy_time = benchmark([&sample, &dest] {
                    cuda::memcpy(dest, sample.data());
                });
                std::cout << "copy time = " << to_milliseconds(copy_time).count() << "ms\n";
            }
        }
    }

    std::cout << "pinned memory:\n";
    {
        std::vector<T, cuda::pinned_allocator<T>> sample;
        auto allocation_time = benchmark([&sample] {
            sample.resize(size);
        });
        std::cout << "allocation time = " << to_milliseconds(allocation_time).count() << "ms\n";

        switch (test_fill) {
            case true: {
                auto fill_time = benchmark([&sample] {
                    std::fill(std::begin(sample), std::end(sample), T(100.0));
                });
                std::cout << "fill time = " << to_milliseconds(fill_time).count() << "ms\n";
                break;
            }
            case false: {
                cuda::managed_ptr<T> dest(size);
                auto copy_time = benchmark([&sample, &dest] {
                    cuda::memcpy(dest, sample.data());
                });
                std::cout << "copy time = " << to_milliseconds(copy_time).count() << "ms\n";
            }
        }
    }
}

void test_stream_matrix_add() {
    using T = float;

    constexpr int n = 1 << 13, size = n * n;

    cuda::stream stream;

    cuda::common_data<T> lhs(stream, size), rhs(stream, size), result(stream, size);
    random_fill(std::begin(lhs), std::end(lhs));
    random_fill(std::begin(rhs), std::end(rhs));

    std::vector<T> cpu_result(size);
    auto cpu_time = benchmark([&lhs, &rhs, &cpu_result] {
        cpu::matrix_add(lhs.get_host_readonly(), rhs.get_host_readonly(), &cpu_result[0], n);
    });
    std::cout << "CPU Time: " << to_milliseconds(cpu_time).count() << "ms" << std::endl;

    auto gpu_h2d_time = benchmark([&lhs, &rhs] {
        lhs.copy_to_device();
        rhs.copy_to_device();
        lhs.synchronize();
        rhs.synchronize();
    });
    std::cout << "GPU H2D Transfer Time: " << to_milliseconds(gpu_h2d_time).count() << "ms" << std::endl;

    auto gpu_time = benchmark([&lhs, &rhs, &result, &stream] {
        auto policy = cuda::make_optimal_policy(gpu::matrix_add<T>, 0, stream);
        cuda::launch_kernel(gpu::matrix_add<T>, policy, 
                            lhs.get_device_readonly(), rhs.get_device_readonly(),
                            result.get_device_writeable(), n);
        stream.synchronize();
    });
    std::cout << "GPU Computation Time: " << to_milliseconds(gpu_time).count() << "ms" << std::endl;

    auto gpu_d2h_time = benchmark([&result] {
        result.copy_to_host();
        result.synchronize();
    });
    std::cout << "GPU D2H Transfer Time: " << to_milliseconds(gpu_d2h_time).count() << "ms" << std::endl;

    auto pr = check_result(std::begin(cpu_result), std::end(cpu_result), std::begin(result), T(0.02));
    auto match = (pr.first == std::end(cpu_result));
    std::cout << "CPU and GPU output " << (match ? "match" : "do not match") << std::endl;
    if (!match) {
        std::cout << std::setprecision(std::numeric_limits<T>::digits10 + 1);
        std::cout << "Mismatch: " << *pr.first << " " << *pr.second << std::endl;
    }
}

void test_stream_parallelism() {
    using T = float;

    constexpr int n = 1 << 13, size = n * n;
    constexpr int count = 2;

    std::vector<cuda::stream> streams(count);

    std::vector<cuda::common_data<T>> lhs, rhs, result;
    for (auto& stream : streams) {
        lhs.emplace_back(stream, size);
        rhs.emplace_back(stream, size);
        result.emplace_back(stream, size);
    }

    for(auto& v : lhs)
        random_fill(std::begin(v), std::end(v));

    for(auto& v : rhs)
        random_fill(std::begin(v), std::end(v));

    auto gpu_time_serial = benchmark([&streams, &lhs, &rhs, &result]() {
        for (int i = 0; i < count; i++) {
            lhs[i].copy_to_device();
            rhs[i].copy_to_device();

            auto policy = cuda::make_optimal_policy(gpu::matrix_add<T>, 0, streams[i]);
            cuda::launch_kernel(gpu::matrix_add<T>, policy,
                lhs[i].get_device_readonly(), rhs[i].get_device_readonly(),
                result[i].get_device_writeable(), n);

            result[i].copy_to_host();
            streams[i].synchronize();
        }
    });
    std::cout << "GPU Serial Computation Time: " << to_milliseconds(gpu_time_serial).count() << "ms" << std::endl;

    auto gpu_time_parallel = benchmark([&streams, &lhs, &rhs, &result]() {
        for (int i = 0; i < count; i++) {
            /* make host dirty to ensure that copy_to_device makes a copy */
            lhs[i].get_host_writeable();
            rhs[i].get_host_writeable();

            lhs[i].copy_to_device();
            rhs[i].copy_to_device();

            auto policy = cuda::make_optimal_policy(gpu::matrix_add<T>, 0, streams[i]);
            cuda::launch_kernel(gpu::matrix_add<T>, policy,
                lhs[i].get_device_readonly(), rhs[i].get_device_readonly(),
                result[i].get_device_writeable(), n);

            result[i].copy_to_host();
        }
        
        for (auto& stream : streams)
            stream.synchronize();
    });   
    std::cout << "GPU Parallel Computation Time: " << to_milliseconds(gpu_time_parallel).count() << "ms" << std::endl;
}

void test_fc() {
    using T = float;
    dnn::network<T> net;

    
    matrix<T> weights, bias;
    weights.resize(2, 3);
    for (int i = 0; i < weights.get_cols(); i++)
        for (int j = 0; j < weights.get_rows(); j++)
            weights.at(i, j) = static_cast<T>((i + 1) * (j + 1));

    bias.resize(2, 1);
    for (int i = 0; i < weights.get_rows(); i++)
            weights.at(i) = static_cast<T>(i + 1);
    
    /* WEIGHTS
    ** 1 2 3
    ** 2 4 6
    **
    ** BIAS
    ** 1
    ** 2
    */
    
    dnn::layer_params<T> params;
    params.values["num_inputs"] = 3;
    params.values["num_outputs"] = 2;
    params.values["has_bias"] = true;

    params.matrix["weights"] = std::move(weights);
    params.matrix["bias"] = std::move(bias);
    
    net.add_layer(dnn::layer_type::fully_connected, params);

    matrix<T> input, output;
    input.resize(3, 1);
    input.at(0) = 5;
    input.at(1) = 4;
    input.at(2) = 1;

    net.forward(input, output);

    std::cout << output.at(0) << ' ' << output.at(1) << std::endl;
}

int main() {
    int dev = 0;
    cudaDeviceProp properties;
    CHECK_CUDA(cudaGetDeviceProperties(&properties, dev));
    std::cout << "Device: " << dev << ", " << properties.name << '\n' << std::endl;

    CHECK_CUDA(cudaSetDevice(dev));
    CHECK_CUDA(cudaFree(0)); /* establish context beforehand so that the benchmarks are not disturbed */

    std::cout << "DATA TRANSFER:\n";
   // test_data_transfer();
    std::cout << std::endl;

    std::cout << "VECTOR ADDITION:\n";
    //test_vector_add();
    std::cout << std::endl;

    std::cout << "MATRIX ADDITION:\n";
    //test_matrix_add();
    std::cout << std::endl;

    std::cout << "MATRIX ADDITION (using stream)\n";
    //test_stream_matrix_add();
    std::cout << std::endl;

    std::cout << "MATRIX MULTIPLICATION:\n";
    test_matrix_multiply();
    std::cout << std::endl;

    std::cout << "MATRIX ADDITION (stream parallelism)\n";
    test_stream_parallelism();
    std::cout << std::endl;

    std::cout << "FULL CONNECTED LAYER\n";
    test_fc();
    std::cout << std::endl;

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}