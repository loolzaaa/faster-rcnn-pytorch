import torch
import torch.nn as nn
from _C import roi_pool_forward
from _C import roi_pool_backward

class _ROIPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, rois, spatial_scale, output_size):
        maxval, argmax = roi_pool_forward(input, rois, spatial_scale, output_size)
            
        ctx.save_for_backward(torch.tensor(input.shape, dtype=torch.int32), argmax, rois)
        
        return maxval

    @staticmethod
    def backward(ctx, grad_output):
        input_size, argmax, rois = ctx.saved_tensors

        grad_input = roi_pool_backward(grad_output, argmax, input_size, rois)
            
        return grad_input, None, None, None


roi_pool = _ROIPool.apply


class ROIPool(nn.Module):
    def __init__(self, spatial_scale, output_size):
        super(ROIPool, self).__init__()
        self.spatial_scale = spatial_scale
        self.output_size = output_size

    def forward(self, input, rois):
        return roi_pool(input, rois, self.spatial_scale, self.output_size)