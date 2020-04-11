import numpy as np
from colorama import Back, Fore
from config import cfg
from dataset import detection_set, pascal_voc

def get_classes(dataset_sequence):
    dataset_name = dataset_sequence.split('_')[0]
    if dataset_name == 'voc':
        year = dataset_sequence.split('_')[1]
        short_name = dataset_name + '_' + year
        return pascal_voc.CLASSES, short_name
    else:
        raise NotImplementedError(Back.RED + 'Not implement for "{:s}" dataset!'.format(dataset_name))

def get_dataset(dataset_sequence, add_params, mode='train'):
    print(Back.WHITE + Fore.BLACK + 'Loading image dataset...')
    dataset_name = dataset_sequence.split('_')[0]
    if dataset_name == 'detect':
        dataset = detection_set.DetectionSet(image_path=add_params['image_path'],
                                             classes=add_params['classes'])
        short_name = 'det_set'
        print('Loaded Detection dataset.')
    elif dataset_name == 'voc':
        year = dataset_sequence.split('_')[1]
        image_set = dataset_sequence[(len(dataset_name) + len(year) + 2):]
        devkit_path = None
        for param in add_params:
            if param.startswith('devkit_path='):
                devkit_path = param.split('=')[1]
        if devkit_path is None:
            print(Back.YELLOW + Fore.BLACK + 'WARNING! ' 
                  + 'Cannot find "devkit_path" in additional parameters. ' 
                  + 'Try to use default path (VOCdevkit)...')
            devkit_path = 'VOCdevkit'
        dataset = pascal_voc.PascalVoc(image_set, year, devkit_path)
        short_name = dataset_name + '_' + year
        print('Loaded PascalVoc {:s} {:s} dataset.'.format(year, image_set))
    else:
        raise NotImplementedError(Back.RED + 'Not implement for "{:s}" dataset!'.format(dataset_name))

    if mode == 'train' and cfg.TRAIN.USE_FLIPPED:
        print(Back.WHITE + Fore.BLACK + 'Appending horizontally-flipped ' 
                                      + 'training examples...')
        dataset = _append_flipped_images(dataset)
        print('Done.')

    print(Back.WHITE + Fore.BLACK + 'Preparing image data...')
    dataset = _prepare_data(dataset)
    print('Done.')

    if mode == 'train':
        print(Back.WHITE + Fore.BLACK + 'Filtering image data ' 
                                      + '(remove images without boxes)...')
        dataset = _filter_data(dataset)
        print('Done.')

    return dataset, short_name

def _append_flipped_images(dataset):
    for i in range(len(dataset)):
        img = dataset.image_data[i].copy()
        img['index'] = len(dataset)
        img['id'] += '_f'
        img['flipped'] = True
        boxes = img['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = img['width'] - oldx2 - 1
        boxes[:, 2] = img['width'] - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        img['boxes'] = boxes
        dataset.image_data.append(img)
        dataset._image_index.append(img['id'])

    return dataset

def _prepare_data(dataset):
    for i in range(len(dataset)):
        # TODO: is this really need!?
        # max overlap with gt over classes (columns)
        max_overlaps = dataset.image_data[i]['gt_overlaps'].max(axis=1)
        # gt class that had the max overlap
        max_classes = dataset.image_data[i]['gt_overlaps'].argmax(axis=1)
        dataset.image_data[i]['max_classes'] = max_classes
        dataset.image_data[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

    return dataset

def _filter_data(dataset):
    print('Before filtering, there are %d images...' % (len(dataset)))
    i = 0
    while i < len(dataset):
        if len(dataset.image_data[i]['boxes']) == 0:
            del dataset.image_data[i]
            i -= 1
        i += 1

    print('After filtering, there are %d images...' % (len(dataset)))
    return dataset


