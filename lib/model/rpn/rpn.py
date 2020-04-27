import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from model.rpn.proposal_layer import _ProposalLayer
from model.rpn.anchor_target_layer import _AnchorTargetLayer
from utils.net_utils import smooth_l1_loss

class _RPN(nn.Module):
    def __init__(self, in_depth):
        super(_RPN, self).__init__()
        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(in_depth, in_depth, 3, 1, 1, bias=True)
                                      
        _anchors_per_point = len(cfg.RPN.ANCHOR_SCALES) * len(cfg.RPN.ANCHOR_RATIOS)
        
        # define bg/fg classifcation score layer
        self.nc_score_out = _anchors_per_point * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(in_depth, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = _anchors_per_point * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(in_depth, self.nc_bbox_out, 1, 1, 0)
        
        # define proposal layer
        self.RPN_proposal = _ProposalLayer(_anchors_per_point)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(_anchors_per_point)
        
    def forward(self, base_feature, im_info, gt_boxes):
        rpn_conv = F.relu(self.RPN_Conv(base_feature), inplace=True)
        
        rpn_cls_score = self.RPN_cls_score(rpn_conv) # B x A*2 x H x W
        _size = rpn_cls_score.size(3) # save width of feature map
        rpn_cls_score_reshape = rpn_cls_score.view(rpn_cls_score.size(0),
                                                   2,
                                                   -1,
                                                   _size)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)
        rpn_cls_prob = rpn_cls_prob_reshape.view(rpn_cls_prob_reshape.size(0),
                                                 self.nc_score_out,
                                                 -1,
                                                 _size)
        
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv) # B x A*4 x H x W
        
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data, im_info, cfg_key))
        
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        
        if self.training:
            rpn_data = self.RPN_anchor_target((rpn_cls_score.size(), gt_boxes, im_info))
            rpn_label, bbox_targets = rpn_data
            
            rpn_keep_all = rpn_label.view(-1).ne(-1).nonzero().view(-1)
            rpn_keep_pos = rpn_label.view(-1).eq(1).nonzero().view(-1)
            
            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
            rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep_all)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep_all).long()
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)

            # compute bounding box loss
            rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous().view(-1, 4)
            rpn_bbox_pred = torch.index_select(rpn_bbox_pred, 0, rpn_keep_pos)
            bbox_targets = torch.index_select(bbox_targets.view(-1, 4), 0, rpn_keep_pos)
            self.rpn_loss_box = smooth_l1_loss(rpn_bbox_pred,
                                               bbox_targets,
                                               beta=1./9,
                                               theta=1./rpn_keep_pos.numel(),
                                               size_average=False)
            self.rpn_loss_box = self.rpn_loss_box / rpn_keep_all.numel()
            
        return rois, self.rpn_loss_cls, self.rpn_loss_box