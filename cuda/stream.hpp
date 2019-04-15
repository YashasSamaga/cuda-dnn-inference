#ifndef CUDA_STREAM_HPP
#define CUDA_STREAM_HPP

#include <cuda_runtime.h>

#include "error.hpp"
#include "../utils/noncopyable.hpp"

namespace cuda {
    class stream : noncopyable {
    public:
        using flags_t = unsigned int;
        using priority_t = int;

        stream() { CHECK_CUDA(cudaStreamCreate(&s)); }
        stream(flags_t flags) { CHECK_CUDA(cudaStreamCreateWithFlags(&s, flags)); }
        stream(flags_t flags, priority_t priority) { CHECK_CUDA(cudaStreamCreateWithPriority(&s, flags, priority)); }
        ~stream() { CHECK_CUDA(cudaStreamDestroy(s)); }

        stream& operator=(stream&&) = default;

        cudaStream_t get() const noexcept { return s; }

        flags_t get_flags() const {
            flags_t flags;
            CHECK_CUDA(cudaStreamGetFlags(s, &flags));
            return flags;
        }

        priority_t get_priority() const {
            priority_t priority;
            CHECK_CUDA(cudaStreamGetPriority(s, &priority));
            return priority;
        }

        void synchronize() { CHECK_CUDA(cudaStreamSynchronize(s)); }

        bool busy() {
            auto status = cudaStreamQuery(s);
            if (status == cudaErrorNotReady)
                return true;
            CHECK_CUDA(status);
            return false;
        }

        operator cudaStream_t() const noexcept { return s; }

    private:
        cudaStream_t s;
    };
}
#endif /* CUDA_STREAM_HPP */