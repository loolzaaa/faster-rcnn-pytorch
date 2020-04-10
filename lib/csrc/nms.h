#pragma once

#include "cuda/header.h"

torch::Tensor nms(const torch::Tensor& proposals, 
                  const torch::Tensor& scores, 
                  const float threshold) {
    if (proposals.numel() == 0) {
        return torch::empty({0}, proposals.options().dtype(at::kLong));
    }
    return nms_cuda(proposals, scores, threshold);
}