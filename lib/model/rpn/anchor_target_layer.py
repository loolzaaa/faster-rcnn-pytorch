import torch
import torch.nn as nn
from config import cfg
from model.rpn.anchor_generate import generate
from utils.bbox_transform import bbox_overlaps_batch, bbox_transform_batch

class _AnchorTargetLayer(nn.Module):
    def __init__(self, anchors_per_point):
        super(_AnchorTargetLayer, self).__init__()
        self._anchors_per_point = anchors_per_point
        
    def forward(self, input):
        height, width = input[0][2], input[0][3]
        gt_boxes = input[1]
        im_info = input[2]

        batch_size = gt_boxes.size(0)
        
        # Generate default anchors with shape: shifts X base_anchors X 4
        anchors = generate(feature_h=height, feature_w=width)
        anchors = anchors.view(-1, 4).to(gt_boxes)
        
        total_anchors = int(anchors.size(0))
        
        keep = ((anchors[:, 0] >= 0) &
                (anchors[:, 1] >= 0) &
                (anchors[:, 2] < int(im_info[0][1])) &
                (anchors[:, 3] < int(im_info[0][0])))
        
        idx_inside = torch.nonzero(keep).view(-1)
        # keep only inside anchors
        anchors = anchors[idx_inside, :]
        
        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = gt_boxes.new_full((batch_size, idx_inside.size(0)), -1)
        bbox_inside_weights = gt_boxes.new_zeros((batch_size, idx_inside.size(0)))
        bbox_outside_weights = gt_boxes.new_zeros((batch_size, idx_inside.size(0)))

        # Overlaps (IoU) size: batch_size x anchors x gt_boxes
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)
        
        # Find max/argmax IoU for every anchor (between anchor and every gt_box)
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2) # batch_size x anchors
        
        # Find absolutely max IoU for every gt_box
        gt_max_overlaps, _ = torch.max(overlaps, 1)
        
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        
        gt_max_overlaps[gt_max_overlaps == 0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)
        
        if torch.sum(keep) > 0:
            labels[keep > 0] = 1
            
        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
        
        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        
        num_fg = int(cfg.TRAIN.RPN_FG_LABELS_FRACTION * cfg.TRAIN.RPN_MAX_LABELS)

        sum_fg = torch.sum((labels == 1).long(), 1)
        sum_bg = torch.sum((labels == 0).long(), 1)
        
        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                rand_num = torch.randperm(fg_inds.size(0)).to(gt_boxes.device, torch.long)
                disable_inds = fg_inds[rand_num[:fg_inds.size(0) - num_fg]]
                labels[i][disable_inds] = -1

            num_bg = cfg.TRAIN.RPN_MAX_LABELS - torch.sum((labels == 1).long(), 1)[i]

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                rand_num = torch.randperm(bg_inds.size(0)).to(gt_boxes.device, torch.long)
                disable_inds = bg_inds[rand_num[:bg_inds.size(0) - num_bg]]
                labels[i][disable_inds] = -1
                
        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        
        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).to(argmax_overlaps)
        bbox_targets = self._compute_targets_batch(anchors, gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))
        
        # use a single value instead of 4 values for easy index.
        bbox_inside_weights[labels == 1] = 1.0
        
        num_examples = torch.sum(labels[i] >= 0)
        positive_weights = 1.0 / num_examples.item()
        negative_weights = 1.0 / num_examples.item()
        
        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights
        
        labels = self._unmap(labels, total_anchors, idx_inside, batch_size, fill=-1)
        bbox_targets = self._unmap(bbox_targets, total_anchors, idx_inside, batch_size, fill=0)
        bbox_inside_weights = self._unmap(bbox_inside_weights, total_anchors, idx_inside, batch_size, fill=0)
        bbox_outside_weights = self._unmap(bbox_outside_weights, total_anchors, idx_inside, batch_size, fill=0)

        outputs = []

        labels = labels.view(batch_size, height, width, self._anchors_per_point).permute(0,3,1,2).contiguous()
        labels = labels.view(batch_size, 1, self._anchors_per_point * height, width)
        outputs.append(labels)

        bbox_targets = bbox_targets.view(batch_size, height, width, self._anchors_per_point*4).permute(0,3,1,2).contiguous()
        outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*self._anchors_per_point)\
                            .permute(0,3,1,2).contiguous()

        outputs.append(bbox_inside_weights)
        
        bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*self._anchors_per_point)\
                            .permute(0,3,1,2).contiguous()
        outputs.append(bbox_outside_weights)
        
        return outputs
        
    def _unmap(self, data, count, inds, batch_size, fill=0):
        """ Unmap a subset of item (data) back to the original set of items (of
        size count) """

        if data.dim() == 2:
            ret = data.new_full((batch_size, count), fill)
            ret[:, inds] = data
        else:
            ret = data.new_full((batch_size, count, data.size(2)), fill)
            ret[:, inds,:] = data
        return ret
        
    def _compute_targets_batch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])