#pragma once

#include "cpu/header.h"

#ifdef WITH_CUDA
#include "cuda/header.h"
#endif

std::tuple<torch::Tensor, torch::Tensor> roi_pool_forward(const torch::Tensor& input, 
                                                          const torch::Tensor& rois, 
                                                          const float spatial_scale,
                                                          const int output_size) {
    if (input.is_cuda()) {
#if defined(WITH_CUDA)
        return roi_pool_forward_cuda(input, rois, spatial_scale, output_size);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return roi_pool_forward_cpu(input, rois, spatial_scale, output_size);
}

torch::Tensor roi_pool_backward(const torch::Tensor& grad_output,
                                const torch::Tensor& argmax,
                                const torch::Tensor& input_size,
                                const torch::Tensor& rois) {
    if (grad_output.is_cuda()) {
#if defined(WITH_CUDA)
        return roi_pool_backward_cuda(grad_output, argmax, input_size, rois);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return roi_pool_backward_cpu(grad_output, argmax, input_size, rois);
}