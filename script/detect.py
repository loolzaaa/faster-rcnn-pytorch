import os
import sys
import pprint
import numpy as np
import time
import cv2 as cv
import torch
import dataset.dataset_factory as dataset_factory
from colorama import Back, Fore
from config import cfg
from torch.utils.data import DataLoader
from dataset.collate import collate_test
from model.vgg16 import VGG16
from model.resnet import Resnet
from utils.bbox_transform import bbox_transform_inv, clip_boxes
from utils.net_utils import vis_detections
from _C import nms

def detect(dataset, net, class_agnostic, load_dir, session, epoch, vis, 
           image_dir, add_params):
    device = torch.device('cuda:0') if cfg.CUDA else torch.device('cpu')
    print(Back.CYAN + Fore.BLACK + 'Current device: %s' % (str(device).upper()))

    print(Back.WHITE + Fore.BLACK + 'Using config:')
    print('GENERAL:')
    pprint.pprint(cfg.GENERAL)
    print('TEST:')
    pprint.pprint(cfg.TEST)
    print('RPN:')
    pprint.pp(cfg.RPN)

    image_dir = os.path.join(cfg.DATA_DIR, image_dir)

    classes, ds_name = dataset_factory.get_classes(dataset)
    add_params['image_path'] = image_dir
    add_params['classes'] = classes
    dataset, _ = dataset_factory.get_dataset('detect', add_params, mode='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, 
                        collate_fn=collate_test)

    if 'data_path' in add_params: cfg.DATA_DIR = add_params['data_path']
    output_dir = os.path.abspath(os.path.join(image_dir, 'result'))
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

    faster_rcnn.eval()

    for idx, img in enumerate(loader):
        total_tic = time.time()
        im2show = cv.imread(dataset.image_path_at(img[3][0]))
        det_tic = time.time()
        rois, cls_prob, bbox_pred, *_ = faster_rcnn(img[0].to(device), img[1].to(device), None)
        
        boxes = rois.data[:, :, 1:5]
        scores = cls_prob.data

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            # Optionally normalize targets by a precomputed mean and stdev
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                box_deltas = box_deltas.view(-1, 4) * torch.Tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(device) \
                            + torch.Tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(device)
                if class_agnostic:
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(1, -1, 4 * len(classes))
            
            pred_boxes = bbox_transform_inv(boxes, box_deltas)
            pred_boxes = clip_boxes(pred_boxes, img[1].data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = torch.repeat_interleave(boxes, dataset.num_classes, 2)

        pred_boxes /= img[1][0][2]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        for j in range(1, len(classes)):
            inds = torch.nonzero(scores[:, j] > 0.05).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, dim=0, descending=True)
                if class_agnostic:
                    cls_boxes = pred_boxes[inds, :].contiguous()
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4].contiguous()
            
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes, cls_scores, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, classes[j], cls_dets.cpu().numpy(), 0.5)

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('\rim_detect: {:d}/{:d} {:.3f}s {:.3f}s ' \
                        .format(idx + 1, len(dataset), detect_time, nms_time))
        sys.stdout.flush()

        if vis:
            result_path = os.path.join(image_dir, 'result', img[3][0] + '_det.jpg')
            cv.imwrite(result_path, im2show)
        else:
            cv.imshow("frame", im2show)
            total_toc = time.time()
            total_time = total_toc - total_tic
            frame_rate = 1 / total_time
            print('Frame rate:', frame_rate)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
    print(Back.GREEN + Fore.BLACK + 'Detection complete.')