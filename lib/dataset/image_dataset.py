import os
import pickle
from config import cfg
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, name, params):
        self._name = name
        self._classes = []
        self._image_index = []
        self._image_data = None

        color_mode = params['color_mode'].upper() if 'color_mode' in params else 'BGR'
        assert color_mode == 'RGB' or color_mode == 'BGR', \
            'Allowed only RGB or BGR color mode. Actual: {}'.format(color_mode)

        range = 255 if 'image_range' not in params else params['image_range']
        assert range == 1 or range == 255, \
            'Allowed only 1 or 255 image range. Actual: {}'.format(range)
        
        mean = [102.9801,
                115.9465,
                122.7717] if 'mean' not in params else params['mean']
        std = [1.0, 1.0, 1.0] if 'std' not in params else params['std']

        self._config = {'color_mode': color_mode,
                        'range': range,
                        'mean': mean,
                        'std': std}
        print('Used image config: ', end='')
        print(self._config)

    def __getitem__(self, index):
        return self._image_data[index]

    def __len__(self):
        return len(self._image_data)

    @property
    def name(self):
        return self._name

    @property
    def classes(self):
        return self._classes

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def image_index(self):
        return self._image_index

    @property
    def image_data(self):
        return self._image_data

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    def _load_image_data(self):
        image_data = []
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                image_data = pickle.load(fid)
            print('Data for {} gt roidb loaded from {}'.format(self.name, cache_file))
        else:
            image_data = [self._load_annotation(idx, id) for idx, id in enumerate(self.image_index)]

            with open(cache_file, 'wb') as fid:
                pickle.dump(image_data, fid, pickle.HIGHEST_PROTOCOL)
            print('Wrote gt roidb to {}'.format(cache_file))

        for img in image_data:
            for k, v in self._config.items():
                img[k] = v

        return image_data

    def image_path_at(self, id):
        raise NotImplementedError

    def _load_annotation(self, idx, id):
        raise NotImplementedError
