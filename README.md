# A PyTorch implementation of Faster R-CNN
Repo based on [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0) (PyTorch 1.0 version)

## Tutorial:
[Blog](http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns) by [ankur6ue](https://github.com/ankur6ue)

## TODOs:
1. Fix README
2. Some cleanup code
3. Add some dataset support to net
4. ...

## Usage:
1. clone repo
2. go to `lib` folder and run `python setup.py develop`
3. make `data` directory in repo root
4. make `pretrained_model` directory in data dir
5. download pretrained [VGG16](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0) or [Resnet101](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0) model and put to step 4 folder
**NOTE:** file name of pretrained model must be <net_name>.pth, where <net_name> may be "vgg16", "resnet#".
6. prepare dataset as described [here](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)
7. run `python run.py -h`

## Prerequisites:
- Python 3.8.0
- PyTorch 1.4.0
- CUDA 10
- Other libraries in [this gist (Conda environment)](https://gist.github.com/loolzaaa/fdbc406d281db9dc0a723536a41679d6)
