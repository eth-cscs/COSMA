#pragma once

#include "util.hpp"
#include "cuda_event.hpp"

// wrapper around cuda streams
class cuda_stream {
  public:
    cuda_stream():
        stream_(new_stream())
    {};

    ~cuda_stream() {
        auto status = cudaStreamDestroy(stream_);
        cuda_check_status(status);
    }

    // return the CUDA stream handle
    cudaStream_t stream() {
        return stream_;
    }

    // insert event into stream
    // returns immediately
    cuda_event enqueue_event() {
        cuda_event e;

        auto status = cudaEventRecord(e.event(), stream_);
        cuda_check_status(status);

        return e;
    }

    // make all future work on stream wait until event has completed.
    // returns immediately, not waiting for event to complete
    void wait_on_event(cuda_event &e) {
        auto status = cudaStreamWaitEvent(stream_, e.event(), 0);
        cuda_check_status(status);
    }

  private:

    cudaStream_t new_stream() {
        cudaStream_t s;

        auto status = cudaStreamCreate(&s);
        cuda_check_status(status);

        return s;
    }

    cudaStream_t stream_;
};
