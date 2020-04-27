import torch
import torch.nn as nn
from config import cfg
from utils.bbox_transform import bbox_overlaps_batch, bbox_transform_batch

class _ProposalTargetLayer(nn.Module):
    def __init__(self, regression_weights):
        super(_ProposalTargetLayer, self).__init__()
        self._regression_weights = regression_weights
        
    def forward(self, all_rois, gt_boxes):
        gt_boxes_append = gt_boxes.new_zeros((gt_boxes.size()))
        gt_boxes_append[:, :, 1:5] =  gt_boxes[:, :, :4]
        
        # Include ground-truth boxes in the set of candidate rois
        all_rois = torch.cat((all_rois, gt_boxes_append), 1)
        
        rois_per_image = cfg.TRAIN.PROPOSAL_PER_IMG
        fg_rois_per_image = int(torch.round(torch.Tensor([rois_per_image])
                                * cfg.TRAIN.FG_PROPOSAL_FRACTION))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image
        
        labels, rois, bbox_targets = \
            self._sample_rois_pytorch(all_rois, gt_boxes, fg_rois_per_image, rois_per_image)

        return rois, labels, bbox_targets

    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        
        # overlaps: (batch_size x rois x gt_boxes)
        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)
        
        max_overlaps, gt_assignment = torch.max(overlaps, 2)
        
        batch_size = overlaps.size(0)
        
        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        offset = offset.view(-1, 1).to(gt_assignment) + gt_assignment
        
        labels = gt_boxes[:,:,4].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)
        
        labels_batch = labels.new_zeros((batch_size, rois_per_image))
        rois_batch  = all_rois.new_zeros((batch_size, rois_per_image, 5))
        gt_rois_batch = all_rois.new_zeros((batch_size, rois_per_image, 5))
        
        for i in range(batch_size):
            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESHOLD).view(-1)
            fg_num_rois = fg_inds.numel()
            
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESHOLD_HI) &
                                    (max_overlaps[i] >= cfg.TRAIN.BG_THRESHOLD_LO)).view(-1)
            bg_num_rois = bg_inds.numel()
            
            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                
                rand_num = torch.randperm(fg_num_rois).to(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
                
                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
                
                rand_num = torch.floor(torch.rand(bg_rois_per_this_image).to(gt_boxes)
                                        * bg_num_rois).long()
                bg_inds = bg_inds[rand_num]                
            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                rand_num = torch.floor(torch.rand(rois_per_image).to(gt_boxes)
                                        * fg_num_rois).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                rand_num = torch.floor(torch.rand(rois_per_image).to(gt_boxes)
                                        * bg_num_rois).long()
                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")
                
            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)
            
            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])
            
            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0
                
            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i,:,0] = i
            
            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]
            
        bbox_target_data = self._compute_targets_pytorch(rois_batch[:,:,1:5], gt_rois_batch[:,:,:4])
        
        bbox_targets = \
                self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch)
        
        return labels_batch, rois_batch, bbox_targets
    
    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        targets = bbox_transform_batch(ex_rois, gt_rois, self._regression_weights)

        return targets
    
    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)
        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).
        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new_zeros((batch_size, rois_per_image, 4))

        for b in range(batch_size):
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]

        return bbox_targets