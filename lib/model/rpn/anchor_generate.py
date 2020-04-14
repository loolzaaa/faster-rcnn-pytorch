import torch
from config import cfg

def generate(feature_h, feature_w, base_size=16):
    ratios = torch.Tensor(cfg.RPN.ANCHOR_RATIOS)
    scales = torch.Tensor(cfg.RPN.ANCHOR_SCALES)
    feature_stride = cfg.RPN.FEATURE_STRIDE
    
    base_anchor = torch.Tensor([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = torch.cat([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
                         
    shift_x = torch.arange(0, feature_w) * feature_stride
    shift_y = torch.arange(0, feature_h) * feature_stride
    shift_x, shift_y = torch.meshgrid(shift_x, shift_y)
    shifts = torch.cat((shift_x.t().contiguous().view(1, -1), 
                        shift_y.t().contiguous().view(1, -1), 
                        shift_x.t().contiguous().view(1, -1), 
                        shift_y.t().contiguous().view(1, -1)))
    shifts = shifts.t().contiguous().float()
    
    A = anchors.size(0)
    K = shifts.size(0)

    anchors = anchors.to(shifts.dtype)
    anchors = anchors.view(1, A, 4) + shifts.view(K, 1, 4)
    
    return anchors
    
def _anchor_params(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr
    
def _make_anchors(ws, hs, x_ctr, y_ctr):
    ws.unsqueeze_(1)
    hs.unsqueeze_(1)
    return torch.cat((x_ctr - 0.5 * (ws - 1),
                      y_ctr - 0.5 * (hs - 1),
                      x_ctr + 0.5 * (ws - 1),
                      y_ctr + 0.5 * (hs - 1)), 1)
    
def _ratio_enum(anchor, ratios):
    w, h, x_ctr, y_ctr = _anchor_params(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = torch.round(torch.sqrt(size_ratios))
    hs = torch.round(ws * ratios)
    return _make_anchors(ws, hs, x_ctr, y_ctr)
    
def _scale_enum(anchor, scales):
    w, h, x_ctr, y_ctr = _anchor_params(anchor)
    ws = w * scales
    hs = h * scales
    return _make_anchors(ws, hs, x_ctr, y_ctr)