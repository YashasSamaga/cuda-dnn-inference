#ifndef CUDA_WORKSPACE_HPP
#define CUDA_WORKSPACE_HPP

#include <cstddef>
#include "memory.hpp"

namespace cuda {
    /* reserves regions of memory as scratchpad
    ** can be shared by multiple layers as long as they don't execute simultaneously
    */
    class workspace {
    public:
        void require(std::size_t bytes) {
            if (bytes > ptr.size())
                ptr.reset(bytes);
        }

        std::size_t size() { return ptr.size(); };

        auto get() { return ptr.get(); }

    private:
        managed_ptr<unsigned char> ptr;
    };
}

#endif /* CUDA_WORKSPACE_HPP */