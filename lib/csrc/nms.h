#pragma once

#include "cpu/header.h"

#ifdef WITH_CUDA
#include "cuda/header.h"
#endif

torch::Tensor nms(const torch::Tensor& proposals, 
                  const torch::Tensor& scores, 
                  const float threshold) {
    if (proposals.is_cuda()) {
#if defined(WITH_CUDA)
        if (proposals.numel() == 0) {
            return torch::empty({0}, proposals.options().dtype(torch::kLong));
        }
        return nms_cuda(proposals, scores, threshold);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return nms_cpu(proposals, scores, threshold);
}