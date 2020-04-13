#pragma once

#include <torch/extension.h>

torch::Tensor nms_cpu(const at::Tensor& proposals,
                      const at::Tensor& scores,
                      const float threshold);

std::tuple<torch::Tensor, torch::Tensor> roi_pool_forward_cpu(const torch::Tensor input,
                                                              const torch::Tensor rois, 
                                                              const float spatial_scale, 
                                                              const int output_size);
                                                               
torch::Tensor roi_pool_backward_cpu(const torch::Tensor grad_output,
                                    const torch::Tensor argmax,
                                    const torch::Tensor input_size,
                                    const torch::Tensor rois);