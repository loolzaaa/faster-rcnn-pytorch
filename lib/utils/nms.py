import torch
from config import cfg
from _C import nms as nms_gpu

def nms(proposals, scores, threshold):
    if cfg.CUDA:
        return nms_gpu(proposals, scores, threshold)
    else:
        return nms_cpu(proposals, scores, threshold)

def nms_cpu(proposals, scores, threshold):
    x1 = proposals[:,0]
    y1 = proposals[:,1]
    x2 = proposals[:,2]
    y2 = proposals[:,3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    _, order = torch.sort(scores, dim=0, descending=True)
    
    num_proposals = proposals.size(0)
    
    suppressed = torch.zeros(num_proposals)
    
    for i in range(num_proposals):
        if suppressed[order[i]] == 1:
            continue
        
        xx1 = torch.max(x1[order[i]], x1[order[i+1:]])
        yy1 = torch.max(y1[order[i]], y1[order[i+1:]])
        xx2 = torch.min(x2[order[i]], x2[order[i+1:]])
        yy2 = torch.min(y2[order[i]], y2[order[i+1:]])
        
        _null_t = torch.tensor((0.0)).type_as(proposals)
        w = torch.max(_null_t, xx2 - xx1 + 1)
        h = torch.max(_null_t, yy2 - yy1 + 1)
        intersection = w * h
        intersection_over_union = intersection / (areas[order[i]] + areas[order[i+1:]] - intersection)
        
        idx = torch.nonzero(intersection_over_union > threshold).view(-1)
        suppressed[idx + i + 1] = 1
        
    return torch.nonzero(suppressed == 0).squeeze(1)
