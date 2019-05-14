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
#include <cmath>

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
        cuda::blas::gemm<T>(handle, false, false, n, n, n, 1.0f, second, n, first, n, 0.0, result, n);
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

template <class T>
void print_matrix2d(const matrix<T>& mat) {
    for (std::size_t i = 0; i < mat.get_height(); i++) {
        for (std::size_t j = 0; j < mat.get_width(); j++) {
            std::cout << mat.at(i, j) << 
                ' ' << i << ' ' << j << '\n';
        }
        std::cout << '\n';
    }
}

void test_fully_connected() {
    using T = float;

    constexpr std::size_t in_width = 3;
    constexpr std::size_t in_height = 3;
    constexpr std::size_t in_chans = 1;
    constexpr std::size_t in_samples = 1;

    constexpr std::size_t outputs = 2;

    matrix<T> fc_weights, fc_bias;
    std::cout << "fc weight matrix:\n";
    fc_weights.resize(outputs, in_width * in_height * in_chans);
    for (int i = 0; i < fc_weights.get_height(); i++)
        for (int j = 0; j < fc_weights.get_width(); j++)
            fc_weights.at(i, j) = T((i + 1) * (j + 1));
    print_matrix2d(fc_weights);

    std::cout << "fc1 bias matrix:\n";
    fc_bias.resize(outputs, 1);
    for (int i = 0; i < fc_bias.get_height(); i++)
        fc_bias.at(i, 0) = T(i + 1);
    print_matrix2d(fc_bias);

    dnn::layer_params<T> params;
    params.matrix["weights"] = std::move(fc_weights);
    params.matrix["bias"] = std::move(fc_bias);

    dnn::network<T> net;
    net.add_layer(dnn::layer_type::fully_connected, params);

    std::cout << "Input Matrix:\n";
    matrix<T> input, output;
    input.resize(in_samples, in_chans, in_height, in_width);
    for (int n = 0; n < input.get_n(); n++)
        for (int c = 0; c < input.get_chans(); c++)
            for (int i = 0; i < input.get_height(); i++)
                for (int j = 0; j < input.get_width(); j++)
                    input.at(n, c, i, j) = T(i + 1);
    print_matrix2d(input);

    input.reshape(input.get_n(), 1, input.size() / input.get_n(), 1);
    net.forward(input, output);

    std::cout << "Output Matrix:\n";
    output.reshape(in_samples, 1, output.get_chans(), 1);
    print_matrix2d(output);

    assert(output.at(0, 0) == 109 && output.at(1, 0) == 218);
}

void test_softmax() {
    using T = float;

    constexpr std::size_t channels = 5;

    dnn::layer_params<T> params;

    dnn::network<T> net;
    net.add_layer(dnn::layer_type::softmax, params);

    std::cout << "Input Matrix:\n";
    matrix<T> input, output;
    input.resize(1, 1, channels, 1); /* chans and height flipped for printing purpose */
    for (int n = 0; n < input.get_n(); n++)
        for (int c = 0; c < input.get_chans(); c++)
            for (int i = 0; i < input.get_height(); i++)
                for (int j = 0; j < input.get_width(); j++)
                    input.at(n, c, i, j) = T(i + 1);
    print_matrix2d(input);

    input.reshape(1, channels, 1, 1);
    net.forward(input, output);

    std::cout << "Output Matrix:\n";
    output.reshape(1, 1, channels, 1);
    print_matrix2d(output);
}

void test_convolution_layer() {
    using T = float;

    constexpr std::size_t in_width = 3;
    constexpr std::size_t in_height = 3;
    constexpr std::size_t in_chans = 1;
    constexpr std::size_t in_samples = 1;

    constexpr std::size_t filters = 1;
    constexpr std::size_t ker_height = 2;
    constexpr std::size_t ker_width = 2;

    matrix<T> conv1_weights, conv1_bias;
    std::cout << "conv1 weight matrix:\n";
    conv1_weights.resize(filters, in_chans, ker_height, ker_width);
    for (int n = 0; n < conv1_weights.get_n(); n++)
        for (int c = 0; c < conv1_weights.get_chans(); c++)
            for (int i = 0; i < conv1_weights.get_height(); i++)
                for (int j = 0; j < conv1_weights.get_width(); j++)
                    conv1_weights.at(i, j) = T(std::floor(std::exp(i * 2 + j + 1)));
    print_matrix2d(conv1_weights);

    std::cout << "conv1 bias matrix:\n";
    conv1_bias.resize(filters, 1);
    for (int i = 0; i < conv1_bias.get_height(); i++)
        conv1_bias.at(i, 0) = T(i + 1);
    print_matrix2d(conv1_bias);

    conv1_bias.reshape(1, filters, 1, 1);

    dnn::layer_params<T> params;
    params.matrix["weights"] = std::move(conv1_weights);
    params.matrix["bias"] = std::move(conv1_bias);

    dnn::network<T> net;
    net.add_layer(dnn::layer_type::convolution, params);

    std::cout << "Input Matrix:\n";
    matrix<T> input, output;
    input.resize(in_samples, in_chans, in_height, in_width);
    for (int n = 0; n < input.get_n(); n++)
        for (int c = 0; c < input.get_chans(); c++)
            for (int i = 0; i < input.get_height(); i++)
                for (int j = 0; j < input.get_width(); j++)
                    input.at(n, c, i, j) = T(i * -j + 1);
    print_matrix2d(input);

    net.forward(input, output);

    std::cout << "Output Matrix:\n";
    print_matrix2d(output);
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
    //test_matrix_multiply();
    std::cout << std::endl;

    std::cout << "MATRIX ADDITION (stream parallelism)\n";
   // test_stream_parallelism();
    std::cout << std::endl;

    std::cout << "FULLY CONNECTED LAYER\n";
    test_fully_connected();
    std::cout << std::endl;

    std::cout << "SOFTMAX LAYER\n";
    test_softmax();
    std::cout << std::endl;

    std::cout << "CONVOLUTION LAYER\n";
    test_convolution_layer();
    std::cout << std::endl;

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}