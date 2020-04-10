import torch
from torch import nn
from torch.autograd import Function
from config import cfg
from _C import roi_pool_forward as roi_pool_forward_cuda
from _C import roi_pool_backward as roi_pool_backward_cuda

# TODO: Fix roi pooling for CPU. Actual work is incorrect ! ! !

class _ROIPool(Function):
    @staticmethod
    def forward(ctx, input, rois, spatial_scale, output_size):
        if cfg.CUDA:
            maxval, argmax = roi_pool_forward_cuda(input, rois, spatial_scale, output_size)
            
            ctx.save_for_backward(torch.tensor(input.shape, dtype=torch.int32), argmax, rois)
            
            return maxval
        else:
            batch_size = input.shape[0]
            channels = input.shape[1]
            height = input.shape[2]
            width = input.shape[3]
            pooling_width = output_size
            pooling_height = output_size

            rois = rois.clone().float()
            rois[:, 1:] = torch.round(rois[:, 1:] * spatial_scale)

            # N = number of ROIs
            # P = Pool area (pooling_width * pooling_height)
            # If width and height of roi are negative, force to 1
            roi_w = torch.clamp(rois[:, 3::5] - rois[:, 1::5] + 1, min=1)
            roi_h = torch.clamp(rois[:, 4::5] - rois[:, 2::5] + 1, min=1)
            # Calculate width and height of one pool segment
            bin_w = roi_w / pooling_width
            bin_h = roi_h / pooling_height
            # Base multipliers of (start_w, start_h, end_w, end_h) for every roi
            # Size: N x 1 x 4
            bin_grid = torch.cat((bin_w, bin_h, bin_w, bin_h), 1).view(-1, 1, 4)

            # Base grid of (start_w, start_h, end_w, end_h) for every point of roi
            # Size: 1 x P x 4
            mw = torch.arange(pooling_width)
            mh = torch.arange(1, pooling_height + 1)
            grid_w, grid_h = torch.meshgrid(mw, mh)
            grid = torch.cat((grid_w.t().contiguous().view(-1, 1), 
                            grid_w.contiguous().view(-1, 1), 
                            grid_h.contiguous().view(-1, 1), 
                            grid_h.t().contiguous().view(-1, 1)), 1).view(1, -1, 4)

            # Full roi grid of (start_w, start_h, end_w, end_h) + clip to image
            # Size: N x P x 4
            roi_grid = bin_grid * grid.type_as(bin_grid)
            roi_grid = torch.cat((torch.floor(roi_grid[:, :, :2]), 
                                torch.ceil(roi_grid[:, :, 2:])), 2)
            roi_grid[:, :, 0] = torch.clamp(roi_grid[:, :, 0] + rois[:, 1::5], 0, width)
            roi_grid[:, :, 1] = torch.clamp(roi_grid[:, :, 1] + rois[:, 2::5], 0, height)
            roi_grid[:, :, 2] = torch.clamp(roi_grid[:, :, 2] + rois[:, 1::5], 0, width)
            roi_grid[:, :, 3] = torch.clamp(roi_grid[:, :, 3] + rois[:, 2::5], 0, height)
            roi_grid = roi_grid.long()

            # Output size: N x channels x P x 2
            output = torch.zeros(roi_grid.shape[0], channels, roi_grid.shape[1], 2).type_as(input)
            for i in range(batch_size):
                img = input[i] # channels x height x width
                batch_rois = roi_grid[rois[:, 0] == i] # N x P x 4
                for j in range(batch_rois.shape[0]):
                    roi = batch_rois[j] # P x 4
                    for k in range(roi.shape[0]):
                        seg = roi[k] # 4 (start_w, start_h, end_w, end_h)
                        img_seg = img[:, seg[1]:seg[3], seg[0]:seg[2]]
                        if img_seg.numel() == 0:
                            output[batch_rois.shape[0] * i + j, :, k, 1] = -1
                        else:
                            max1, arg1 = torch.max(img_seg, 2)
                            max2, arg2 = torch.max(max1, 1)
                            arg = img.shape[2] * (seg[1] + arg2) + seg[0] + arg1[torch.arange(channels), arg2]
                            output[batch_rois.shape[0] * i + j, :, k, 0] = max2
                            output[batch_rois.shape[0] * i + j, :, k, 1] = arg

            ctx.save_for_backward(torch.tensor(input.shape, dtype=torch.int32), output[:, :, :, 1], rois)
            
            return output[:, :, :, 0].view(-1, channels, pooling_height, pooling_width)

    @staticmethod
    def backward(ctx, grad_output):
        input_size, argmax, rois = ctx.saved_tensors
        
        # N = number of ROIs
        if cfg.CUDA:
            # Argmax size: N x channels x pooling_height x pooling_width
            grad_input = roi_pool_backward_cuda(grad_output, argmax, input_size, rois)
            
            return grad_input, None, None, None
        else:
            # P = Pool area (pooling_width * pooling_height)
            # Argmax size: N x channels x P            
            batch_size = input_size[0]
            channels = input_size[1]
            pool_area = argmax.shape[2]

            grad_output = grad_output.view(-1, channels, pool_area)
            grad_input = torch.zeros(*input_size).view(batch_size, channels, -1).type_as(grad_output)
            argmax = argmax.view(batch_size, -1, channels, pool_area)
            for i in range(batch_size):
                arg_batch = argmax[i] # N/batch_size x channels x P
                for j in range(arg_batch.shape[0]):
                    roi = arg_batch[j] # channels x P
                    for k in range(roi.shape[1]):
                        arg = roi[:, k].long() # channels
                        if roi.sum() > 0:
                            grad_input[i, torch.arange(channels), arg] += grad_output[i * arg_batch.shape[0] + j, :, k]

            return grad_input.view(*input_size), None, None, None


roi_pool = _ROIPool.apply


class ROIPool(nn.Module):
    def __init__(self, spatial_scale, output_size):
        super(ROIPool, self).__init__()
        self.spatial_scale = spatial_scale
        self.output_size = output_size

    def forward(self, input, rois):
        return roi_pool(input, rois, self.spatial_scale, self.output_size)