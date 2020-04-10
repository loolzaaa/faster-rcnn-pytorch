#pragma once

#include "cuda/header.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor> roi_pool_forward(const torch::Tensor& input, 
                                                          const torch::Tensor& rois, 
                                                          const float spatial_scale,
                                                          const int output_size) {
    CHECK_INPUT(input);
    CHECK_INPUT(rois);
    return roi_pool_forward_cuda(input, rois, spatial_scale, output_size);
}

torch::Tensor roi_pool_backward(const torch::Tensor& grad_output,
                                const torch::Tensor& argmax,
                                const torch::Tensor& input_size,
                                const torch::Tensor& rois) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(argmax);
    CHECK_INPUT(rois);
    //AT_ASSERTM(input_size.size()[0] == 4, "Input size is incorrect!");
    return roi_pool_backward_cuda(grad_output, argmax, input_size, rois);
}