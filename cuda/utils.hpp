#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include "error.hpp"
#include "stream.hpp"

#include <cuda_runtime.h>

namespace cuda {
    struct execution_policy {
        execution_policy(dim3 grid_size, dim3 block_size) 
            : grid{ grid_size }, block{ block_size }, sharedMem{ 0 }, stream{ 0 } { }

        execution_policy(dim3 grid_size, dim3 block_size, std::size_t shared_mem)
            : grid{ grid_size }, block{ block_size }, sharedMem{ shared_mem }, stream{ 0 } { }

        execution_policy(dim3 grid_size, dim3 block_size, cudaStream_t strm)
            : grid{ grid_size }, block{ block_size }, sharedMem{ 0 }, stream{ strm } { }

        execution_policy(dim3 grid_size, dim3 block_size, std::size_t shared_mem, cudaStream_t strm)
            : grid{ grid_size }, block{ block_size }, sharedMem{ shared_mem }, stream{ strm } { }

        dim3 grid;
        dim3 block;
        std::size_t sharedMem;
        cudaStream_t stream;
    };

    template <class Kernel> inline
    execution_policy make_optimal_policy(Kernel kernel, std::size_t sharedMem = 0, cudaStream_t stream = 0) {
        int grid_size, block_size;
        CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, kernel, sharedMem));
        assert(block_size % 32 == 0);
        return execution_policy(grid_size, block_size, sharedMem, stream);
    }

    template <class Kernel, typename ...Args> inline
    void launch_kernel(Kernel kernel, Args ...args) {
        auto policy = make_optimal_policy(kernel);
        kernel <<<policy.grid, policy.block>>> (std::forward<Args>(args)...);
    }

    template <class Kernel, typename ...Args> inline
    void launch_kernel(Kernel kernel, dim3 grid, dim3 block, Args ...args) {
        kernel <<<grid, block>>> (std::forward<Args>(args)...);
    }

    template <class Kernel, typename ...Args> inline
    void launch_kernel(Kernel kernel, execution_policy policy, Args ...args) {
        kernel <<<policy.grid, policy.block, policy.sharedMem, policy.stream>>> (std::forward<Args>(args)...);
    }

    inline void device_synchronize() { CHECK_CUDA(cudaDeviceSynchronize()); }

    template <int> __device__ auto getGridDim()->decltype(dim3::x);
    template <> inline __device__ auto getGridDim<0>()->decltype(dim3::x) { return gridDim.x; }
    template <> inline __device__ auto getGridDim<1>()->decltype(dim3::x) { return gridDim.y; }
    template <> inline __device__ auto getGridDim<2>()->decltype(dim3::x) { return gridDim.z; }

    template <int> __device__ auto getBlockDim()->decltype(dim3::x);
    template <> inline __device__ auto getBlockDim<0>()->decltype(dim3::x) { return blockDim.x; }
    template <> inline __device__ auto getBlockDim<1>()->decltype(dim3::x) { return blockDim.y; }
    template <> inline __device__ auto getBlockDim<2>()->decltype(dim3::x) { return blockDim.z; }

    template <int> __device__ auto getBlockIdx()->decltype(uint3::x);
    template <> inline __device__ auto getBlockIdx<0>()->decltype(uint3::x) { return blockIdx.x; }
    template <> inline __device__ auto getBlockIdx<1>()->decltype(uint3::x) { return blockIdx.y; }
    template <> inline __device__ auto getBlockIdx<2>()->decltype(uint3::x) { return blockIdx.z; }

    template <int> __device__ auto getThreadIdx()->decltype(uint3::x);
    template <> inline __device__ auto getThreadIdx<0>()->decltype(uint3::x) { return threadIdx.x; }
    template <> inline __device__ auto getThreadIdx<1>()->decltype(uint3::x) { return threadIdx.y; }
    template <> inline __device__ auto getThreadIdx<2>()->decltype(uint3::x) { return threadIdx.z; }

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
}

#endif /* CUDA_UTILS_HPP */