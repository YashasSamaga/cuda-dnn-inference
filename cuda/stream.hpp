#ifndef CUDA_STREAM_HPP
#define CUDA_STREAM_HPP

#include "error.hpp"
#include "../utils/noncopyable.hpp"

#include <cuda_runtime.h>
#include <memory>

namespace cuda {
    struct default_stream_t {
        static constexpr cudaStream_t stream = 0;
    };

    class unique_stream : noncopyable {
    public:
        using flags_type = unsigned int;
        using priority_type = int;

        unique_stream() { CHECK_CUDA(cudaStreamCreate(&stream)); }
        unique_stream(default_stream_t) noexcept : stream{ default_stream_t::stream } {}
        unique_stream(unique_stream&&) noexcept = default;
        unique_stream(flags_type flags) { CHECK_CUDA(cudaStreamCreateWithFlags(&stream, flags)); }
        unique_stream(flags_type flags, priority_type priority) { CHECK_CUDA(cudaStreamCreateWithPriority(&stream, flags, priority)); }
        ~unique_stream() { if(stream != default_stream_t::stream) CHECK_CUDA(cudaStreamDestroy(stream)); }

        flags_type get_flags() const {
            flags_type flags;
            CHECK_CUDA(cudaStreamGetFlags(stream, &flags));
            return flags;
        }

        priority_type get_priority() const {
            priority_type priority;
            CHECK_CUDA(cudaStreamGetPriority(stream, &priority));
            return priority;
        }

        void synchronize() { CHECK_CUDA(cudaStreamSynchronize(stream)); }

        bool busy() {
            auto status = cudaStreamQuery(stream);
            if (status == cudaErrorNotReady)
                return true;
            CHECK_CUDA(status);
            return false;
        }

        auto get() const noexcept { return stream; }

        operator cudaStream_t() const noexcept { return stream; }

    private:
        cudaStream_t stream;
    };

    class stream {
    public:
        using flags_type = unique_stream::flags_type;
        using priority_type = unique_stream::priority_type;

        stream() : strm{ std::make_shared<unique_stream>() } {  }
        stream(default_stream_t ds) : strm{ std::make_shared<unique_stream>(ds) } {  }
        stream(stream&) noexcept = default;
        stream(stream&&) noexcept = default;
        stream(unique_stream&& other) : strm{ std::make_shared<unique_stream>(std::move(other)) } { }
        stream(flags_type flags) : strm{ std::make_shared<unique_stream>(flags) } { }
        stream(flags_type flags, priority_type priority) : strm{ std::make_shared<unique_stream>(flags, priority) } { }
        ~stream() { }

        stream& operator=(stream&) noexcept = default;
        stream& operator=(stream&&) noexcept = default;

        cudaStream_t get() const noexcept { return strm->get(); }
        flags_type get_flags() const { return strm->get_flags(); }
        priority_type get_priority() const { return strm->get_priority(); }
        void synchronize() const { strm->synchronize(); }
        bool busy() const { return strm->busy(); }

        operator cudaStream_t() const noexcept { return strm->get(); }

    private:
        std::shared_ptr<unique_stream> strm;
    };
}
#endif /* CUDA_STREAM_HPP */