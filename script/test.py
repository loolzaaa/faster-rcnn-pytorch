import os
import sys
import time
import pprint
import pickle
import torch
import numpy as np
import cv2 as cv
import dataset.dataset_factory as dataset_factory
from colorama import Back, Fore
from config import cfg
from torch.utils.data import DataLoader
from dataset.collate import collate_test
from model.vgg16 import VGG16
from model.resnet import Resnet
from utils.bbox_transform import bbox_transform_inv, clip_boxes
from _C import nms

def test(dataset, net, class_agnostic, load_dir, session, epoch, add_params):
    device = torch.device('cuda:0') if cfg.CUDA else torch.device('cpu')
    print(Back.CYAN + Fore.BLACK + 'Current device: %s' % (str(device).upper()))

    print(Back.WHITE + Fore.BLACK + 'Using config:')
    print('GENERAL:')
    pprint.pprint(cfg.GENERAL)
    print('TEST:')
    pprint.pprint(cfg.TEST)
    print('RPN:')
    pprint.pp(cfg.RPN)

    # TODO: add competition mode
    dataset, ds_name = dataset_factory.get_dataset(dataset, add_params, mode='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, 
                        collate_fn=collate_test)

    if 'data_path' in add_params: cfg.DATA_DIR = add_params['data_path']
    output_dir = os.path.join(cfg.DATA_DIR, 'output', net, ds_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(Back.CYAN + Fore.BLACK + 'Output directory: %s' % (output_dir))

    if net == 'vgg16':
        faster_rcnn = VGG16(dataset.num_classes, class_agnostic=class_agnostic)
    elif net.startswith('resnet'):
        num_layers = net[6:]
        faster_rcnn = Resnet(num_layers, dataset.num_classes, class_agnostic=class_agnostic)
    else:
        raise ValueError(Back.RED + 'Network "{}" is not defined!'.format(net))

    faster_rcnn.init()
    faster_rcnn.to(device)

    model_path = os.path.join(cfg.DATA_DIR, load_dir, net, ds_name, 
                              'frcnn_{}_{}.pth'.format(session, epoch))
    print(Back.WHITE + Fore.BLACK + 'Loading model from %s' % (model_path))
    checkpoint = torch.load(model_path, map_location=device)
    faster_rcnn.load_state_dict(checkpoint['model'])
    print('Done.')

    start = time.time()
    max_per_image = 100
        
    all_boxes = [[[] for _ in range(len(dataset))] for _ in range(dataset.num_classes)]

    faster_rcnn.eval()

    for i, data in enumerate(loader):
        image_data = data[0].to(device)
        image_info = data[1].to(device)

        det_tic = time.time()
        with torch.no_grad():
            rois, cls_prob, bbox_pred, *_ = faster_rcnn(image_data, image_info, None)

        boxes = rois[:, :, 1:5]
        breakpoint()

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Unnormalize targets if it was while training process
                bbox_pred = bbox_pred.view(-1, 4) \
                          * torch.Tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(device) \
                          + torch.Tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(device)
                if class_agnostic:                    
                    bbox_pred = bbox_pred.view(1, -1, 4)
                else:
                    bbox_pred = bbox_pred.view(1, -1, 4 * dataset.num_classes)

            pred_boxes = bbox_transform_inv(boxes, bbox_pred)
            pred_boxes = clip_boxes(pred_boxes, image_info, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = torch.repeat_interleave(boxes, dataset.num_classes, 2)

        pred_boxes /= image_info[0][2].item()

        scores = cls_prob.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic

        misc_tic = time.time()
        for j in range(1, dataset.num_classes):
            inds = torch.nonzero(scores[:,j] > 0).view(-1)
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = torch.empty(0, 5).numpy()

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in range(1, dataset.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, dataset.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
            .format(i + 1, len(dataset), detect_time, nms_time))
        sys.stdout.flush()
            
    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('\nEvaluating detections...')
    dataset.evaluate_detections(all_boxes, output_dir)

    # TODO: Add txt file with result info ?

    end = time.time()
    print(Back.GREEN + Fore.BLACK + 'Test time: %.4fs.' % (end - start))