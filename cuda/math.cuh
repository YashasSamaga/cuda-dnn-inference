#ifndef CUDA_MATH_CUH
#define CUDA_MATH_CUH

#include "utils.hpp"
#include "span.hpp"

namespace cuda {
    namespace math {
        namespace detail {
            template <class T> __device__ T abs(T val);
            template <> inline __device__ float abs(float val) { return fabsf(val); }
            template <> inline __device__ double abs(double val) { return fabs(val); }

            template <class T> __device__ T exp(T val);
            template <> inline __device__ float exp(float val) { return expf(val); }
            template <> inline __device__ double exp(double val) { return exp(val); }

            template <class T> __device__ T max(T x, T y);
            template <> inline __device__ float max(float x, float y) { return fmaxf(x, y); }
            template <> inline __device__ double max(double x, double y) { return fmax(x, y); }

            template <class T> __device__ T min(T x, T y);
            template <> inline __device__ float min(float x, float y) { return fminf(x, y); }
            template <> inline __device__ double min(double x, double y) { return fmin(x, y); }

            template <class T> __device__ T log(T val);
            template <> inline __device__ float log(float val) { return logf(val); }
            template <> inline __device__ double log(double val) { return log(val); }

            template <class T> __device__ T tanh(T val);
            template <> inline __device__ float tanh(float val) { return tanhf(val); }
            template <> inline __device__ double tanh(double val) { return tanh(val); }

            template <class T> __device__ T pow(T val, T exp);
            template <> inline __device__ float pow(float val, float exp) { return powf(val, exp); }
            template <> inline __device__ double pow(double val, double exp) { return pow(val, exp); }

            template <class T>
            __device__ T sigmoid(T val) { return T(1) / (1 + exp(-val)); }
        }

        namespace kernels {
            template <class T>
            __global__ void abs(view<T> src, span<T> dest) {
                assert(src.size() >= dest.size());
                for (auto i : grid_stride_range(dest.size())) {
                    using detail::abs;
                    dest[i] = abs(src[i]);
                }
            }

            template <class T>
            __global__ void tanh(view<T> src, span<T> dest) {
                assert(src.size() >= dest.size());
                for (auto i : grid_stride_range(dest.size())) {
                    using detail::tanh;
                    dest[i] = tanh(src[i]);
                }
            }

            template <class T>
            __global__ void sigmoid(view<T> src, span<T> dest) {
                assert(src.size() >= dest.size());
                for (auto i : grid_stride_range(dest.size())) {
                    using detail::sigmoid;
                    dest[i] = sigmoid(src[i]);
                }
            }

            template <class T>
            __global__ void bnll(view<T> src, span<T> dest) {
                assert(src.size() >= dest.size());
                for (auto i : grid_stride_range(dest.size())) {
                    using detail::log;
                    using detail::exp;
                    dest[i] = log(1 + exp(-src[i]));
                }
            }

            template <class T>
            __global__ void elu(view<T> src, span<T> dest, T alpha) {
                assert(src.size() >= dest.size());
                for (auto i : grid_stride_range(dest.size())) {
                    using detail::exp;
                    dest[i] = src[i] >= 0 ? src[i] : alpha * (exp(src[i]) - 1);
                    /* TODO branch can be eliminated? */
                }
            }

            template <class T>
            __global__ void relu(view<T> src, span<T> dest, T slope) {
                assert(src.size() >= dest.size());
                for (auto i : grid_stride_range(dest.size())) {
                    dest[i] = src[i] >= 0.0 ? src[i] : slope * src[i];
                }
            }

            template <class T>
            __global__ void clipped_relu(view<T> src, span<T> dest, T ceiling, T floor) {
                assert(src.size() >= dest.size());
                assert(floor <= ceiling);
                for (auto i : grid_stride_range(dest.size())) {
                    using detail::max;
                    using detail::min;
                    dest[i] = min(max(src[i], floor), ceiling);
                }
            }

            template <class T>
            __global__ void power(view<T> src, span<T> dest, T exp, T scale, T shift) {
                assert(src.size() >= dest.size());
                for (auto i : grid_stride_range(dest.size())) {
                    using detail::pow;
                    dest[i] = pow(shift + scale * src[i], exp);
                }
            }
        }

        template <class T>
        void abs(view<T> src, span<T> dest) {
            assert(src.size() >= dest.size());
            launch_kernel(kernels::abs<T>, src, dest);
        }

        template <class T>
        void tanh(view<T> src, span<T> dest) {
            assert(src.size() >= dest.size());
            launch_kernel(kernels::tanh<T>, src, dest);
        }

        template <class T>
        void sigmoid(view<T> src, span<T> dest) {
            assert(src.size() >= dest.size());
            launch_kernel(kernels::sigmoid<T>, src, dest);
        }

        template <class T>
        void bnll(view<T> src, span<T> dest) {
            assert(src.size() >= dest.size());
            launch_kernel(kernels::bnll<T>, src, dest);
        }

        template <class T>
        void elu(view<T> src, span<T> dest, T alpha) {
            assert(src.size() >= dest.size());
            launch_kernel(kernels::elu<T>, src, dest, alpha);
        }

        template <class T>
        void relu(view<T> src, span<T> dest, T slope) {
            assert(src.size() >= dest.size());
            launch_kernel(kernels::relu<T>, src, dest, slope);
        }

        template <class T>
        void clipped_relu(view<T> src, span<T> dest, T floor, T ceiling) {
            assert(src.size() >= dest.size());
            assert(floor <= ceiling);
            launch_kernel(kernels::clipped_relu<T>, src, dest, floor, ceiling);
        }

        template <class T>
        void power(view<T> src, span<T> dest, T exp, T scale, T shift) {
            assert(src.size() >= dest.size());
            launch_kernel(kernels::power<T>, src, dest, exp, scale, shift);
        }
    }
}

#endif /* CUDA_MATH_CUH */