#pragma once

#include "cpu/header.h"

#ifdef WITH_CUDA
#include "cuda/header.h"
#endif

torch::Tensor roi_align_forward(
    const torch::Tensor& input, // Input feature map.
    const torch::Tensor& rois, // List of ROIs to pool over.
    const float spatial_scale, // The scale of the image features.
    const int output_size, // The height/width of the pooled feature map.
    const int sampling_ratio, // The number of points to sample in each bin
    const bool aligned) // The flag for pixel shift along each axis.
{
    if (input.is_cuda()) {
#if defined(WITH_CUDA)
        return roi_align_forward_cuda(input, rois, spatial_scale,
            output_size, sampling_ratio, aligned);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return roi_align_forward_cpu(input, rois, spatial_scale,
        output_size, sampling_ratio, aligned);
}

torch::Tensor roi_align_backward(const torch::Tensor& grad,
                                 const torch::Tensor& rois,
                                 const float spatial_scale,
                                 const torch::Tensor& input_size,
                                 const int sampling_ratio,
                                 const bool aligned) {
    if (grad.is_cuda()) {
#if defined(WITH_CUDA)
        return roi_align_backward_cuda(grad, rois, spatial_scale,
            input_size, sampling_ratio, aligned);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return roi_align_backward_cpu(grad, rois, spatial_scale,
        input_size, sampling_ratio, aligned);
}