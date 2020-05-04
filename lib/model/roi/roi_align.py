import torch
import torch.nn as nn
from _C import roi_align_forward
from _C import roi_align_backward

class _ROIAlign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, rois, spatial_scale, output_size, sampling_ratio, aligned):
        """
        Performs Region of Interest (RoI) Align operator described in Mask R-CNN
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
                the box coordinates.
            sampling_ratio (int): number of sampling points in the interpolation grid
                used to compute the output value of each pooled output bin. If > 0,
                then exactly sampling_ratio x sampling_ratio grid points are used. If
                <= 0, then an adaptive number of grid points are used (computed as
                ceil(roi_width / pooled_w), and likewise for height).
            aligned (bool): If False, use the legacy implementation.
                If True, pixel shift it by -0.5 for align more perfectly about two neighboring pixel indices.
                This version in Detectron2
        Returns:
            output (Tensor[K, C, output_size[0], output_size[1]])
        """
        result = roi_align_forward(input, rois, spatial_scale, output_size, sampling_ratio, aligned)
            
        ctx.save_for_backward(torch.tensor(input.shape, dtype=torch.int32),
                              rois, torch.Tensor([spatial_scale, sampling_ratio, aligned]))
        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input_size, rois, params = ctx.saved_tensors
        spatial_scale = params[0].item()
        sampling_ratio = params[1].int().item()
        aligned = bool(params[2].int().item())

        grad_input = roi_align_backward(grad_output, rois, spatial_scale,
                                        input_size, sampling_ratio, aligned)
            
        return grad_input, None, None, None, None, None


roi_align = _ROIAlign.apply


class ROIAlign(nn.Module):
    def __init__(self, spatial_scale, output_size, sampling_ratio, aligned=False):
        super(ROIAlign, self).__init__()
        self.spatial_scale = spatial_scale
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input, rois):
        return roi_align(input,
                         rois,
                         self.spatial_scale,
                         self.output_size,
                         self.sampling_ratio,
                         self.aligned)