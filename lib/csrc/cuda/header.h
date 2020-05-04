#pragma once

#include <torch/extension.h>

torch::Tensor nms_cuda(const torch::Tensor& proposals,
                       const torch::Tensor& scores,
                       const float threshold);

std::tuple<torch::Tensor, torch::Tensor> roi_pool_forward_cuda(const torch::Tensor& input,
                                                               const torch::Tensor& rois,
                                                               const float spatial_scale,
                                                               const int output_size);
                                                               
torch::Tensor roi_pool_backward_cuda(const torch::Tensor& grad_output,
                                     const torch::Tensor& argmax,
                                     const torch::Tensor& input_size,
                                     const torch::Tensor& rois);

torch::Tensor roi_align_forward_cuda(const torch::Tensor& input,
                                     const torch::Tensor& rois,
                                     const float spatial_scale,
                                     const int output_size,
                                     const int sampling_ratio,
                                     const bool aligned);

torch::Tensor roi_align_backward_cuda(const torch::Tensor& grad,
                                      const torch::Tensor& rois,
                                      const float spatial_scale,
                                      const torch::Tensor& input_size,
                                      const int sampling_ratio,
                                      const bool aligned);
