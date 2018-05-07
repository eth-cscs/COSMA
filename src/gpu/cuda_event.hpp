#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "util.hpp"

// wrapper around cuda events
class cuda_event {
  public:

    cuda_event() {
        auto status = cudaEventCreate(&event_);
        cuda_check_status(status);
    }

    ~cuda_event() {
        // note that cudaEventDestroy can be called on an event before is has
        // been reached in a stream, and the CUDA runtime will defer clean up
        // of the event until it has been completed.
        auto status = cudaEventDestroy(event_);
        cuda_check_status(status);
    }

    // return the underlying event handle
    cudaEvent_t& event() {
        return event_;
    }

    // force host execution to wait for event completion
    void wait() {
        auto status = cudaEventSynchronize(event_);
        cuda_check_status(status);
    }

    // returns time in seconds taken between this cuda event and another cuda event
    double time_since(cuda_event& other) {
        float time_taken = 0.0f;

        auto status = cudaEventElapsedTime(&time_taken, other.event(), event_);
        cuda_check_status(status);
        return double(time_taken/1.e3);
    }

  private:

    cudaEvent_t event_;
};

