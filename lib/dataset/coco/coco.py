import os
import uuid
import json
import pickle
import numpy as np
import dataset.utils as utils
from dataset.image_dataset import ImageDataset

# COCO API
from dataset.coco.pycocotools.coco import COCO as MSCOCO
from dataset.coco.pycocotools.cocoeval import COCOeval

class COCO(ImageDataset):
    def __init__(self, image_set, year, data_path, only_classes=False):
        ImageDataset.__init__(self, 'COCO_' + year + '_' + image_set)
        self._image_set = image_set
        self._year = year
        self._data_path = data_path
        assert os.path.exists(self._data_path), \
            'Path to data does not exist: {}'.format(self._data_path)
        self._COCO = MSCOCO(self._get_ann_file())
        cats = self._COCO.loadCats(self._COCO.getCatIds())
        self._classes = tuple(['__background__'] + [c['name'] for c in cats])
        if not only_classes:
            self._class_index = dict(zip(self.classes, range(self.num_classes)))
            self._class_coco_index = dict(zip(self.classes[1:], self._COCO.getCatIds()))
            self._image_index = self._COCO.getImgIds()
            self._image_data = self._load_image_data()

            self._view_map = {
                'minival2014': 'val2014',  # 5k val2014 subset
                'valminusminival2014': 'val2014',  # val2014 \setminus minival2014
                'test-dev2015': 'test2015',
                'valminuscapval2014': 'val2014',
                'capval2014': 'val2014',
                'captest2014': 'val2014'
            }
            coco_name = self._image_set + self._year  # e.g., "val2014"
            self._data_name = (self._view_map[coco_name]
                               if coco_name in self._view_map
                               else coco_name)

            # COCO specific config options
            self.config = {'use_salt': True,
                           'cleanup': True}

    def image_path_at(self, id):
        file_name = ('COCO_' + self._data_name + '_' + str(id).zfill(12) + '.jpg')
        image_path = os.path.join(self._data_path, 'images', 
                                  self._data_name, file_name)
        assert os.path.exists(image_path), \
            'Image Path does not exist: {}'.format(image_path)
        return image_path
    
    def _get_ann_file(self):
        prefix = 'instances' if self._image_set.find('test') == -1 else 'image_info'
        return os.path.join(self._data_path, 'annotations', 
                            prefix + '_' + self._image_set + self._year + '.json')

    def _load_annotation(self, idx, id):
        img_path = self.image_path_at(id)
        im_ann = self._COCO.loadImgs(id)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self._COCO.getAnnIds(imgIds=id, iscrowd=None)
        objs = self._COCO.loadAnns(annIds)

        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        areas = np.zeros((num_objs), dtype=np.float32)

        # Map COCO category indexes to our internal class indexes
        coco_index_to_class_index = dict([(self._class_coco_index[cls],
                                           self._class_index[cls])
                                          for cls in self.classes[1:]])

        for ix, obj in enumerate(objs):
            cls = coco_index_to_class_index[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            areas[ix] = obj['area']
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                overlaps[ix, :] = -1.0
            else:
                overlaps[ix, cls] = 1.0

        utils.validate_boxes(boxes, width=width, height=height)
        return {'index': idx,
                'id': id,
                'path': img_path,
                'width': width,
                'height': height,
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'gt_areas': areas,
                'flipped': False}

    def _print_detection_eval_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
               '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
        print('{:.1f}'.format(100 * ap_default))
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print('{:.1f}'.format(100 * ap))

        print('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()

    def _do_detection_eval(self, res_file, output_dir):
        ann_type = 'bbox'
        coco_dt = self._COCO.loadRes(res_file)
        coco_eval = COCOeval(self._COCO, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._print_detection_eval_metrics(coco_eval)
        eval_file = os.path.join(output_dir, 'detection_results.pkl')
        with open(eval_file, 'wb') as fid:
            pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote COCO eval results to: {}'.format(eval_file))

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.image_index):
            dets = boxes[im_ind].astype(np.float)
            if dets == []:
                    continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
                [{'image_id': index,
                  'category_id': cat_id,
                  'bbox': [xs[k], ys[k], ws[k], hs[k]],
                  'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def _write_coco_results_file(self, all_boxes, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                             self.num_classes - 1))
            coco_cat_id = self._class_to_coco_cat_id[cls]
            results.extend(self._coco_results_one_category(all_boxes[cls_ind],
                                                           coco_cat_id))
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = os.path.join(output_dir, ('detections_' +
                                             self._image_set +
                                             self._year +
                                             '_results'))
        if self.config['use_salt']:
            res_file += '_{}'.format(str(uuid.uuid4()))
        res_file += '.json'
        self._write_coco_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        if self._image_set.find('test') == -1:
            self._do_detection_eval(res_file, output_dir)
        # Optionally cleanup results json file
        if self.config['cleanup']:
            os.remove(res_file)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True