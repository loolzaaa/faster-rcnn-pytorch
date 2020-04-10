import os
import numpy as np
from dataset.image_dataset import ImageDataset

class DetectionSet(ImageDataset):
    def __init__(self, image_path, classes):
        ImageDataset.__init__(self, 'Detection dataset')
        self._image_path = image_path
        assert os.path.exists(self._image_path), \
            'Path to data does not exist: {}'.format(self._image_path)
        self._classes = classes
        self._image_index = self._load_image_index()
        self._image_data = self._load_image_data()

    def image_path_at(self, name):
        image_path = os.path.join(self._image_path, name + '.jpg')
        assert os.path.exists(image_path), \
            'Image Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_index(self):
        image_index = []
        for f in os.listdir(self._image_path):
            if f[-3:] == 'jpg' and os.path.isfile(os.path.join(self._image_path, f)):
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
