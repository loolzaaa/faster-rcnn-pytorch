# A PyTorch implementation of Faster R-CNN
Repo based on [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0) (PyTorch 1.0 version)

### ***Only CUDA version works***

## Tutorial:
[Blog]() by [ankur6ue](https://github.com/ankur6ue)

## TODOs:
1. Fix README
2. Some cleanup code
3. Add some backbones to net
4. Add some dataset support to net
5. Add more ROI layers
6. Fix ROI Pool layer (CPU imlementation) - doesn't work now!
7. add requirements file
8. ...

## Usage:
1. clone repo
2. go to `lib` folder and run `python setup.py develop`
3. make `data` directory in repo root
4. make `pretrained_model` directory in data dir
5. [download](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0) pretrained VGG16 model and put to step 3 folder
6. prepare dataset as described [here](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)
7. run `python run.py -h`

-------------------------
- Used PyTorch 1.4.0
- Used Python 3.8.0
- Used CUDA 10
- Used library: pytorch, numpy, opencv, easydict, matplotlib, colorama.
