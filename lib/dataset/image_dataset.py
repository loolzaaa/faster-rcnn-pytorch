import os
from config import cfg
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, name):
        self._name = name
        self._classes = []
        self._image_index = []
        self._image_data = None
        self._config = {}
        
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

    def image_path_at(self, id):
        raise NotImplementedError
