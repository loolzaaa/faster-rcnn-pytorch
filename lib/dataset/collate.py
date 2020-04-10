import cv2 as cv
import numpy as np
import torch
from config import cfg

def collate_train(image_batch):
    return collate(image_batch, is_training=True)

def collate_test(image_batch):
    return collate(image_batch, is_training=False)

def collate(image_batch, is_training):
    batch = []
    for img in image_batch:
        blob = get_image_blob(img)
        data = torch.from_numpy(blob['data'])
        im_info = torch.from_numpy(blob['im_info'])
        img_id = blob['img_id']
        if is_training:
            np.random.shuffle(blob['gt_boxes'])
            gt_boxes = torch.from_numpy(blob['gt_boxes'])
            data, im_info, gt_boxes = crop_image(data, im_info, gt_boxes, img_id)
        else:
            gt_boxes = torch.Tensor([[1,1,1,1,1]])
        
        batch.append((data, im_info, gt_boxes, img_id))

    if len(image_batch) > 1:
        batch = padding_batch(batch)
    
    prepared_batch = prepare_batch(batch)
    return prepared_batch

def get_image_blob(img):
    im = cv.imread(img['path']) #BGR
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.concatenate((im, im, im), axis=2)
    im = im[:, :, ::-1] # BGR -> RGB
    if img['flipped']:
        im = im[:, ::-1, :]
    im = im.astype(np.float32, copy=True)
    im -= np.array([[cfg.GENERAL.PIXEL_MEANS]])
    im_shape = im.shape # H x W x C
    im_size_min = np.min(im_shape[:2])
    im_size_max = np.max(im_shape[:2])
    im_scale = float(600) / im_size_min
    if im_scale * im_size_max > 1000:
        im_scale = float(1000) / im_size_max
    im = cv.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv.INTER_LINEAR)
    
    blob = {'data': im}
    gt_inds = np.where(img['gt_classes'] != 0)[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = img['boxes'][gt_inds, :] * im_scale
    gt_boxes[:, 4] = img['gt_classes'][gt_inds]
    blob['gt_boxes'] = gt_boxes
    blob['im_info'] = np.array([im.shape[0], im.shape[1], im_scale], dtype=np.float32)
    blob['img_id'] = img['id']

    return blob

def crop_image(image_data, image_info, gt_boxes, img_id):
    width = image_data.size(1)
    height = image_data.size(0)
    ratio = float(width) / height
    if ratio < cfg.GENERAL.MIN_IMG_RATIO:
        # image width << image height => crop height
        min_gt_y = int(torch.min(gt_boxes[:, 1]))
        max_gt_y = int(torch.max(gt_boxes[:, 3]))
        max_gt_height = max_gt_y - min_gt_y + 1
        trim_size = int(float(width) / 0.5)
        if max_gt_height >= trim_size:
            y_start_offset = int((max_gt_height - trim_size) / 2)
            y_start = np.random.choice(range(min_gt_y, min_gt_y + y_start_offset + 1))
        else:
            y_start_offset = int((trim_size - max_gt_height) / 2)
            y_start = min_gt_y - y_start_offset
            if y_start < 0:
                y_start = 0

        # crop the image
        image_data = image_data[y_start:(y_start + trim_size), :, :]

        #shift y coordinate of gt boxes
        gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_start)
        gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_start)

        # update gt bounding box according the trip
        gt_boxes[:, 1].clamp_(0, trim_size - 1)
        gt_boxes[:, 3].clamp_(0, trim_size - 1)

        # update image info
        image_info[0] = image_data.size(0)

        # check the bounding box:
        not_keep = ((gt_boxes[:,0] == gt_boxes[:,2]) | 
                    (gt_boxes[:,1] == gt_boxes[:,3]))
        keep = torch.nonzero(not_keep == 0).view(-1)

        if keep.numel() != 0:
            gt_boxes = gt_boxes[keep]

        print('Crop image "%s": %.0fx%.0f -> %.0fx%.0f' \
              % (img_id, width, height, image_data.shape[1], image_data.shape[0]))
        return image_data, image_info, gt_boxes
    elif ratio > cfg.GENERAL.MAX_IMG_RATIO:
        # image width >> image height => crop width
        min_gt_x = int(torch.min(gt_boxes[:, 0]))
        max_gt_x = int(torch.max(gt_boxes[:, 2]))
        max_gt_width = max_gt_x - min_gt_x + 1
        trim_size = int(float(height) * 2)
        if max_gt_width >= trim_size:
            x_start_offset = int((max_gt_width - trim_size) / 2)
            x_start = np.random.choice(range(min_gt_x, min_gt_x + x_start_offset + 1))
        else:
            x_start_offset = int((trim_size - max_gt_width) / 2)
            x_start = min_gt_x - x_start_offset
            if x_start < 0:
                x_start = 0

        # crop the image
        image_data = image_data[:, x_start:(x_start + trim_size), :]

        #shift y coordinate of gt boxes
        gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_start)
        gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_start)

        # update gt bounding box according the trip
        gt_boxes[:, 0].clamp_(0, trim_size - 1)
        gt_boxes[:, 2].clamp_(0, trim_size - 1)

        # update image info
        image_info[1] = image_data.size(1)

        # check the bounding box:
        not_keep = ((gt_boxes[:,0] == gt_boxes[:,2]) | 
                    (gt_boxes[:,1] == gt_boxes[:,3]))
        keep = torch.nonzero(not_keep == 0).view(-1)

        if keep.numel() != 0:
            gt_boxes = gt_boxes[keep]

        print('Crop image "%s": %.0fx%.0f -> %.0fx%.0f' \
              % (img_id, width, height, image_data.shape[1], image_data.shape[0]))
        return image_data, image_info, gt_boxes
    else:
        return image_data, image_info, gt_boxes

def padding_batch(image_batch):
    max_img_height = int(max([img[1][0] for img in image_batch]))
    max_img_width = int(max([img[1][1] for img in image_batch]))
    ratio = float(max_img_width) / max_img_height
    image_info = torch.Tensor([max_img_height, max_img_width, ratio])
    
    batch = []
    for img in image_batch:
        height = img[0].size(0)
        width = img[0].size(1)
        padding_data = torch.Tensor(max_img_height, max_img_width, 3).zero_()
        padding_data[:height, :width, :] = img[0]
        batch.append((padding_data, image_info, img[2], img[3]))

    return batch

def prepare_batch(image_batch):
    num_images = len(image_batch)
    img_height = int(image_batch[0][1][0])
    img_width = int(image_batch[0][1][1])
    data = torch.Tensor(num_images, 3, img_height, img_width).zero_()
    info = torch.Tensor(num_images, 3).zero_()
    gt_boxes = torch.Tensor(num_images, 20, image_batch[0][2].size(1)).zero_()
    id = []
    for i in range(num_images):
        data[i] = image_batch[i][0].permute(2, 0, 1).contiguous()
        info[i] = image_batch[i][1]
        num_boxes = min(image_batch[i][2].size(0), 20)
        gt_boxes[i, :num_boxes, :] = image_batch[i][2][:num_boxes, :]
        id.append(image_batch[i][3])

    return data, info, gt_boxes, id