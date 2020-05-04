import torch
import torch.nn as nn
from _C import roi_pool_forward
from _C import roi_pool_backward

class _ROIPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, rois, spatial_scale, output_size):
        """
        Performs Region of Interest (RoI) Pool operator described in Fast R-CNN
        Arguments:
            input (Tensor[N, C, H, W]): input tensor
            rois (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
                format where the regions will be taken from. If a single Tensor is passed,
                then the first column should contain the batch index. If a list of Tensors
                is passed, then each Tensor will correspond to the boxes for an element i
                in a batch
            output_size (int or Tuple[int, int]): the size of the output after the cropping
                is performed, as (height, width)
            spatial_scale (float): a scaling factor that maps the input coordinates to
                the box coordinates. Default: 1.0
        Returns:
            output (Tensor[K, C, output_size[0], output_size[1]])
        """
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