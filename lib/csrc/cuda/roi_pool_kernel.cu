#include <torch/extension.h>

template <typename scalar_t>
__global__ void roi_pool_forward_cuda_kernel(const int total_size,
                                             const scalar_t *input,
                                             const scalar_t *rois,
                                             const scalar_t spatial_scale,
                                             const int channels,
                                             const int height,
                                             const int width,
                                             const int pooling_width,
                                             const int pooling_height,
                                             scalar_t *output,
                                             int *argmax) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for (int index = i; index < total_size; index += total_threads) {
        // (N, C, point_h, point_w) is an element in the pooled output
        int roi_point_w = index % pooling_width;
        int roi_point_h = (index / pooling_width) % pooling_height;
        int cur_roi_channel = (index / pooling_width / pooling_height) % channels;
        int cur_roi = index / pooling_width / pooling_height / channels;

        const scalar_t *offset_rois = rois + cur_roi * 5;
        int roi_batch_idx = offset_rois[0];
        int roi_x1 = round(offset_rois[1] * spatial_scale);
        int roi_y1 = round(offset_rois[2] * spatial_scale);
        int roi_x2 = round(offset_rois[3] * spatial_scale);
        int roi_y2 = round(offset_rois[4] * spatial_scale);

        // If width and height of roi are negative, force to 1
        int roi_w = max(roi_x2 - roi_x1 + 1, 1);
        int roi_h = max(roi_y2 - roi_y1 + 1, 1);
        scalar_t bin_size_h = static_cast<scalar_t>(roi_h) 
                                / static_cast<scalar_t>(pooling_height);
        scalar_t bin_size_w = static_cast<scalar_t>(roi_w) 
                                / static_cast<scalar_t>(pooling_width);

        int hstart = static_cast<int>(floor(bin_size_h * static_cast<scalar_t>(roi_point_h)));
        int wstart = static_cast<int>(floor(bin_size_w * static_cast<scalar_t>(roi_point_w)));
        int hend = static_cast<int>(ceil(bin_size_h * static_cast<scalar_t>(roi_point_h + 1)));
        int wend = static_cast<int>(ceil(bin_size_w * static_cast<scalar_t>(roi_point_w + 1)));

        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart + roi_y1, 0), height);
        hend = min(max(hend + roi_y1, 0), height);
        wstart = min(max(wstart + roi_x1, 0), width);
        wend = min(max(wend + roi_x1, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        scalar_t maxval = is_empty ? 0 : -FLT_MAX;
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        int maxidx = -1;
        const scalar_t *offset_data =
            input + (roi_batch_idx * channels + cur_roi_channel) * height * width;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            int bottom_index = h * width + w;
            if (offset_data[bottom_index] > maxval) {
              maxval = offset_data[bottom_index];
              maxidx = bottom_index;
            }
          }
        }
        output[index] = maxval;
        argmax[index] = maxidx;
    }
}

template <typename scalar_t>
__global__ void roi_pool_backward_cuda_kernel(const int total_size,
                                              const scalar_t *grad_output,
                                              const int *argmax,
                                              const int batch_size,
                                              const int channels,
                                              const int height,
                                              const int width,
                                              const int pooling_width,
                                              const int pooling_height,
                                              scalar_t *grad_input,
                                              const scalar_t *rois) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for (int index = i; index < total_size; index += total_threads) {
        // (N, C, point_h, point_w) is an element in the pooled output
        int roi_point_w = index % pooling_width;
        int roi_point_h = (index / pooling_width) % pooling_height;
        int cur_roi_channel = (index / pooling_width / pooling_height) % channels;
        int cur_roi = index / pooling_width / pooling_height / channels;
        
        int output_offset = (cur_roi * channels + cur_roi_channel) * pooling_height * pooling_width;
        const scalar_t *grad_output_offset = grad_output + output_offset;
        const int *argmax_offset = argmax + output_offset;
        
        //int total_rois = total_size / pooling_width / pooling_height / channels;
        //int roi_batch_idx = cur_roi / (total_rois / batch_size);
        //int input_offset = (roi_batch_idx * channels + cur_roi_channel) * height * width;
        //scalar_t *grad_input_offset = grad_input + input_offset;
        const scalar_t *offset_rois = rois + cur_roi * 5;
        int roi_batch_ind = offset_rois[0];
        int input_offset = (roi_batch_ind * channels + cur_roi_channel) * height * width;
        scalar_t *grad_input_offset = grad_input + input_offset;
        
        int argmax_value = argmax_offset[roi_point_h * pooling_width + roi_point_w];
        if (argmax_value != -1) {
            atomicAdd(
                grad_input_offset + argmax_value,
                static_cast<scalar_t>(grad_output_offset[roi_point_h * pooling_width + roi_point_w])
            );
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> roi_pool_forward_cuda(const torch::Tensor input, 
                                                               const torch::Tensor rois, 
                                                               const float spatial_scale,
                                                               const int output_size) {
    const int num_rois = rois.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int pooling_width = output_size;
    const int pooling_height = output_size;
    
    const auto total_size = num_rois * pooling_height * pooling_width * channels;
    
    auto output = torch::empty({num_rois, channels, pooling_height, pooling_width}, 
                               input.options());
    auto argmax = torch::zeros({num_rois, channels, pooling_height, pooling_width}, 
                               input.options().dtype(torch::kInt));
    
    const dim3 blocks(std::min((total_size + 512 - 1) / 512, 4096));
    const dim3 threads(512);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "roi_pool_forward_cuda", ([&] {
        roi_pool_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            total_size,
            input.data<scalar_t>(),
            rois.data<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooling_width,
            pooling_height,
            output.data<scalar_t>(),
            argmax.data<int>());
    }));
    
    return std::make_tuple(output, argmax);
}

torch::Tensor roi_pool_backward_cuda(const torch::Tensor grad_output, 
                                     const torch::Tensor argmax, 
                                     const torch::Tensor input_size,
                                     const torch::Tensor rois) {    
    auto input_size_a = input_size.accessor<int,1>();
    const int batch_size = input_size_a[0];
    const int channels = input_size_a[1];
    const int height = input_size_a[2];
    const int width = input_size_a[3];
    
    const int num_rois = argmax.size(0);
    
    const int pooling_width = argmax.size(3);
    const int pooling_height = argmax.size(2);
    
    const auto total_size = num_rois * pooling_height * pooling_width * channels;
    
    auto grad_input = torch::zeros({batch_size, channels, height, width}, 
                                   grad_output.options());
    
    const dim3 blocks(std::min((total_size + 512 - 1) / 512, 4096));
    const dim3 threads(512);
    
    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "roi_pool_backward_cuda", ([&] {
        roi_pool_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            total_size,
            grad_output.data<scalar_t>(),
            argmax.data<int>(),
            batch_size,
            channels,
            height,
            width,
            pooling_width,
            pooling_height,
            grad_input.data<scalar_t>(),
            rois.data<scalar_t>());
    }));
    
    return grad_input;
}