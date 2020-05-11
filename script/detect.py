import os
import sys
import pprint
import time
import cv2 as cv
import torch
import dataset.dataset_factory as dataset_factory
from colorama import Back, Fore
from config import cfg, update_config_from_file
from torch.utils.data import DataLoader
from dataset.collate import collate_test
from model.vgg16 import VGG16
from model.resnet import Resnet
from utils.net_utils import vis_detections
from _C import nms

def detect(dataset, net, class_agnostic, load_dir, session, epoch, vis, 
           image_dir, add_params):
    device = torch.device('cuda:0') if cfg.CUDA else torch.device('cpu')
    print(Back.CYAN + Fore.BLACK + 'Current device: %s' % (str(device).upper()))

    if 'cfg_file' in add_params:
        update_config_from_file(add_params['cfg_file'])

    print(Back.WHITE + Fore.BLACK + 'Using config:')
    print('GENERAL:')
    pprint.pprint(cfg.GENERAL)
    print('TEST:')
    pprint.pprint(cfg.TEST)
    print('RPN:')
    pprint.pp(cfg.RPN)

    image_dir = os.path.join(cfg.DATA_DIR, image_dir)

    dataset, ds_name = dataset_factory.get_dataset(dataset, add_params, 
                                                   mode='test',
                                                   only_classes=True)
    classes = dataset.classes
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

    for i, data in enumerate(loader):
        image_data = data[0].to(device)
        image_info = data[1].to(device)

        total_tic = time.time()
        im2show = cv.imread(dataset.image_path_at(data[3][0]))
        det_tic = time.time()
        with torch.no_grad():
            cls_score, bbox_pred, *_ = faster_rcnn(image_data, image_info, None)

        bbox_pred /= image_info[0][2].item()

        scores = cls_score.squeeze()
        bbox_pred = bbox_pred.squeeze()
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
                    cls_boxes = bbox_pred[inds, :].contiguous()
                else:
                    cls_boxes = bbox_pred[inds][:, j * 4:(j + 1) * 4].contiguous()
            
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes, cls_scores, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, classes[j], cls_dets.cpu().numpy(), 0.5)

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('\rim_detect: {:d}/{:d} {:.3f}s {:.3f}s ' \
                        .format(i + 1, len(dataset), detect_time, nms_time))
        sys.stdout.flush()

        if vis:
            result_path = os.path.join(image_dir, 'result', data[3][0] + '_det.jpg')
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