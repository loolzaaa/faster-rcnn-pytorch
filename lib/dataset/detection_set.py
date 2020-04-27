import os
import numpy as np
from dataset.image_dataset import ImageDataset


class DetectionSet(ImageDataset):
    def __init__(self, params):
        ImageDataset.__init__(self, 'Detection dataset', params)
        self._image_path = params['image_path']
        assert os.path.exists(self._image_path), \
            'Path to data does not exist: {}'.format(self._image_path)
        self._classes = params['classes']
        self._extension = params['img_ext'] if 'img_ext' in params else '.jpg'
        self._image_index = self._load_image_index()
        self._image_data = self._load_image_data()
        for img in self._image_data:
            for k, v in self._config.items():
                img[k] = v

    def image_path_at(self, name):
        image_path = os.path.join(self._image_path, name + self._extension)
        assert os.path.exists(image_path), \
            'Image Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_index(self):
        image_index = []
        for f in os.listdir(self._image_path):
            is_file = os.path.isfile(os.path.join(self._image_path, f))
            if f[-4:] == self._extension and is_file:
                image_index.append(f[:-4])
        return image_index

    def _load_image_data(self):
        image_data = []
        for idx, img in enumerate(self.image_index):
            img_path = self.image_path_at(img)
            boxes = np.zeros((0, 4), dtype=np.uint16)
            gt_classes = np.zeros((0), dtype=np.int32)
            overlaps = np.zeros((0, self.num_classes), dtype=np.float32)
            image_data.append({'index': idx,
                               'id': img,
                               'path': img_path,
                               'boxes': boxes,
                               'gt_classes': gt_classes,
                               'gt_overlaps': overlaps,
                               'flipped': False})
        return image_data
