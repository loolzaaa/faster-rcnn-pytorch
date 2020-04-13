import torch
import torch.nn as nn
import numpy as np
from config import cfg
from model.rpn.anchor_generate import generate
from utils.bbox_transform import bbox_transform_inv, clip_boxes
from _C import nms

class _ProposalLayer(nn.Module):
    def __init__(self, anchors_per_point):
        super(_ProposalLayer, self).__init__()
        self._anchors_per_point = anchors_per_point
        
    def forward(self, input):
        # the first set of _anchors_per_point are bg probs
        # the second set are the fg probs
        scores = input[0][:, self._anchors_per_point:, :, :]
        bbox_deltas = input[1]
        im_info = input[2]
        key = input[3]
        
        pre_nms_topN  = cfg[key].RPN_PRE_NMS_TOP
        post_nms_topN = cfg[key].RPN_POST_NMS_TOP
        nms_thresh    = cfg[key].RPN_NMS_THRESHOLD
        
        batch_size = bbox_deltas.size(0)
        
        # Generate default anchors with shape: shifts X base_anchors X 4
        anchors = generate(feature_h=scores.size(2), feature_w=scores.size(3))
        anchors = anchors.view(1, -1, 4).expand(batch_size, -1, 4).type_as(bbox_deltas) #Surprise! type_as change device of tensor too.
        
        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        # Same story for the scores:
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)
        
        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)
        
        # Clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info, batch_size)
        
        # Sort all (proposal, score) pairs by score from highest to lowest
        _, order = torch.sort(scores, dim=1, descending=True)
        
        output = scores.new_zeros(batch_size, post_nms_topN, 5)
        for i in range(batch_size):
            proposals_single_img = proposals[i]
            scores_single_img = scores[i]
            order_single_img = order[i]
            
            # Take top pre_nms_topN (e.g. 6000)
            if pre_nms_topN > 0 and pre_nms_topN < scores.numel():
                order_single_img = order_single_img[:pre_nms_topN]
                
            proposals_single_img = proposals_single_img[order_single_img, :]
            scores_single_img = scores_single_img[order_single_img]
            
            # Apply nms (e.g. threshold = 0.7)
            keep_idx_i = nms(proposals_single_img, scores_single_img, nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)
            
            # Take after_nms_topN (e.g. 300)
            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single_img = proposals_single_img[keep_idx_i, :]
            
            # padding 0 at the end.
            num_proposal = proposals_single_img.size(0)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single_img
            
        return output