import torchvision
import torch
import torch.nn as nn
from model.faster_rcnn import FasterRCNN

class VGG16(FasterRCNN):
    def __init__(self, num_classes, class_agnostic, pretrained=False, model_path=None):
        FasterRCNN.__init__(self, num_classes, class_agnostic, 512)
        self.pretrained = pretrained
        self.model_path = model_path

    def _init_modules(self):
        backbone = torchvision.models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s..." % (self.model_path))
            state_dict = torch.load(self.model_path)
            backbone.load_state_dict({k:v for k,v in state_dict.items() if k in backbone.state_dict()})
            print('Done.')
        
        backbone.classifier = nn.Sequential(*list(backbone.classifier._modules.values())[:-1])
        
        # not using the last maxpool layer
        self.RCNN_base = nn.Sequential(*list(backbone.features._modules.values())[:-1])

        self.RCNN_top = backbone.classifier

        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
        
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

        # Fix the layers before conv3 for VGG16:
        for layer in range(10):
            for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    def _feed_pooled_feature_to_top(self, pooled_feature):
        return self.RCNN_top(pooled_feature.view(pooled_feature.size(0), -1))