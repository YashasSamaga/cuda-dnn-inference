#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cuda_runtime.h>
#include "error.hpp"

namespace cuda {
    /* TODO execeution_policy */

    void device_synchronize() { CHECK_CUDA(cudaDeviceSynchronize()); }

    template <class Kernel, typename ...Args>
    void launch_kernel(Kernel kernel, Args ...args) {
        int grid_size, block_size;
        CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, kernel));
        kernel <<<grid_size, block_size >>> (args...);
    }

    template <class Kernel, typename ...Args>
    void launch_kernel(Kernel kernel, dim3 grid, dim3 block, Args ...args) {
        kernel <<<grid, block >>> (args...);
    }

    template <int> __device__ auto getGridDim()->decltype(dim3::x);
    template <> __device__ auto getGridDim<0>()->decltype(dim3::x) { return gridDim.x; }
    template <> __device__ auto getGridDim<1>()->decltype(dim3::x) { return gridDim.y; }
    template <> __device__ auto getGridDim<2>()->decltype(dim3::x) { return gridDim.z; }

    template <int> __device__ auto getBlockDim()->decltype(dim3::x);
    template <> __device__ auto getBlockDim<0>()->decltype(dim3::x) { return blockDim.x; }
    template <> __device__ auto getBlockDim<1>()->decltype(dim3::x) { return blockDim.y; }
    template <> __device__ auto getBlockDim<2>()->decltype(dim3::x) { return blockDim.z; }

    template <int> __device__ auto getBlockIdx()->decltype(uint3::x);
    template <> __device__ auto getBlockIdx<0>()->decltype(uint3::x) { return blockIdx.x; }
    template <> __device__ auto getBlockIdx<1>()->decltype(uint3::x) { return blockIdx.y; }
    template <> __device__ auto getBlockIdx<2>()->decltype(uint3::x) { return blockIdx.z; }

    template <int> __device__ auto getThreadIdx()->decltype(uint3::x);
    template <> __device__ auto getThreadIdx<0>()->decltype(uint3::x) { return threadIdx.x; }
    template <> __device__ auto getThreadIdx<1>()->decltype(uint3::x) { return threadIdx.y; }
    template <> __device__ auto getThreadIdx<2>()->decltype(uint3::x) { return threadIdx.z; }

    template <int dim>
    class grid_stride_range_generic {
    public:
        __device__ grid_stride_range_generic(std::size_t to_) : from(0), to(to_) { }
        __device__ grid_stride_range_generic(std::size_t from_, std::size_t to_) : from(from_), to(to_) { }

        class iterator
        {
        public:
            __device__ iterator(std::size_t pos_) : pos(pos_) {}

            __device__ size_t operator*() const { return pos; }

            __device__ iterator& operator++() {
                pos += getGridDim<dim>() * getBlockDim<dim>();
                return *this;
            }

            __device__ bool operator!=(const iterator& other) const {
                /* NOTE TODO HACK
                ** 'pos' can move in large steps (see operator++)
                ** as the expansion of range for loop uses != as the loop conditioion
                ** operator!= must return false if 'pos' crosses the end
                */
                return pos < other.pos;
            }

        private:
            std::size_t pos;
        };

        __device__ iterator begin() const {
            return iterator(from + getBlockDim<dim>() * getBlockIdx<dim>() + getThreadIdx<dim>());
        }

        __device__ iterator end() const {
            return iterator(to);
        }

    private:
        std::size_t from, to;
    };

    using grid_stride_range_x = grid_stride_range_generic<0>;
    using grid_stride_range_y = grid_stride_range_generic<1>;
    using grid_stride_range_z = grid_stride_range_generic<2>;
    using grid_stride_range = grid_stride_range_x;

    /* TODO THINK remove or fix */
    class grid_stride_range_2d {
    public:
        __device__ grid_stride_range_2d(unsigned end1, unsigned end2)
            : range_begin{ 0, 0 }, range_end{ end1, end2 } { }

        class iterator {
        public:
            __device__ iterator() { }
            __device__ iterator(unsigned x, uint2 range_end) : pos{ x }, range_end(range_end) {}

            __device__ uint2 operator*() const {
                return uint2{ pos / range_end.x, pos % range_end.x };
            }

            __device__ iterator& operator++() {
                pos += (blockDim.y * gridDim.y + threadIdx.y) * blockDim.x * gridDim.x;
                return *this;
            }

            __device__ bool operator!=(const iterator& item) const {
                return pos < item.pos;
            }

        private:
            unsigned pos;
            uint2 range_end;
        };

        __device__ iterator begin() const {
            return iterator((range_begin.y + blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x +
                (range_begin.x + blockIdx.x * blockDim.x + threadIdx.x), range_end);
        }

        __device__ iterator end() const {
            return iterator(range_end.x * range_end.y, range_end);
        }

    private:
        uint2 range_begin;
        uint2 range_end;
    };
}

#endif /* CUDA_UTILS_HPP */