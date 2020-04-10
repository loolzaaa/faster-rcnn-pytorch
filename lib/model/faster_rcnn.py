import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from model.rpn.rpn import _RPN
from model.roi.roi_pool import ROIPool
from model.rpn.proposal_target_layer import _ProposalTargetLayer
from utils.net_utils import _smooth_l1_loss

class FasterRCNN(nn.Module):
    def __init__(self, num_classes, class_agnostic=True, pretrained=False, model_path=None):
        super(FasterRCNN, self).__init__()
        self.n_classes = num_classes
        self.class_agnostic = class_agnostic
        self.pretrained = pretrained
        self.model_path = model_path
        
        # define rpn
        self.RCNN_rpn = _RPN(512)
        self.RCNN_proposal_target = _ProposalTargetLayer()
        
        self.RCNN_roi_pool = ROIPool(1.0/16.0, cfg.GENERAL.POOLING_SIZE)
        
        self._init_modules()
        self._init_weights()
        
    def forward(self, im_data, im_info, gt_boxes):
        if self.training:
            assert gt_boxes is not None
    
        batch_size = im_data.size(0)
        
        base_feature = self.RCNN_base(im_data)
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feature, im_info, gt_boxes)
        
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            
            rois_label = rois_label.view(-1).long()
            rois_target = rois_target.view(-1, rois_target.size(2)).requires_grad_(True)
            rois_inside_ws = rois_inside_ws.view(-1, rois_inside_ws.size(2)).requires_grad_(True)
            rois_outside_ws = rois_outside_ws.view(-1, rois_outside_ws.size(2)).requires_grad_(True)
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            
        pooled_feat = self.RCNN_roi_pool(base_feature, rois.view(-1,5))
        
        # feed pooled features to top model
        pooled_feat = self.RCNN_top(pooled_feat.view(pooled_feat.size(0), -1))
        
        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1)
                                            .expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label
        
    def _init_modules(self):
        backbone = torchvision.models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s..." % (self.model_path))
            state_dict = torch.load(self.model_path)
            backbone.load_state_dict({k:v for k,v in state_dict.items() if k in backbone.state_dict()})
            print('Done.')
        
        backbone.classifier = nn.Sequential(*list(backbone.classifier._modules.values())[:-1])
        
        # not using the last maxpool layer
        self.RCNN_base = nn.Sequential(*list(backbone.features._modules.values())[:-1])
        
        # Fix the layers before conv3 for VGG16:
        for layer in range(10):
            for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

        self.RCNN_top = backbone.classifier

        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
        
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)
        
    def _init_weights(self):
        def normal_init(m, mean, stddev):
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01)
        normal_init(self.RCNN_cls_score, 0, 0.01)
        normal_init(self.RCNN_bbox_pred, 0, 0.001)
