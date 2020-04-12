import torchvision
import torch
import torch.nn as nn
from config import cfg
from model.faster_rcnn import FasterRCNN

resnet_vars = {
    'resnet18': [torchvision.models.resnet18, 256],
    'resnet34': [torchvision.models.resnet34, 256],
    'resnet50': [torchvision.models.resnet50, 1024],
    'resnet101': [torchvision.models.resnet101, 1024],
    'resnet152': [torchvision.models.resnet152, 1024],
}

class Resnet(FasterRCNN):
    def __init__(self, num_layers, num_classes, class_agnostic=True, 
                 pretrained=False, model_path=None):
        FasterRCNN.__init__(self, num_classes, class_agnostic, resnet_vars['resnet' + num_layers][1])
        self.num_layers = num_layers
        self.pretrained = pretrained
        self.model_path = model_path

    def _init_modules(self):
        backbone = resnet_vars['resnet' + self.num_layers][0]()
        if self.pretrained:
            print("Loading pretrained weights from %s..." % (self.model_path))
            state_dict = torch.load(self.model_path)
            backbone.load_state_dict({k:v for k,v in state_dict.items() if k in backbone.state_dict()})
            print('Done.')

        self.RCNN_base = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                       backbone.maxpool, backbone.layer1, 
                                       backbone.layer2, backbone.layer3)

        self.RCNN_top = nn.Sequential(backbone.layer4)

        self.RCNN_cls_score = nn.Linear(backbone.fc.in_features, self.n_classes)
        
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(backbone.fc.in_features, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(backbone.fc.in_features, 4 * self.n_classes)

        # Fix blocks
        for p in self.RCNN_base[0].parameters(): p.requires_grad=False # conv1
        for p in self.RCNN_base[1].parameters(): p.requires_grad=False # bn1

        assert (0 <= cfg.RESNET.NUM_FREEZE_BLOCKS < 4)
        for i in range(4, cfg.RESNET.NUM_FREEZE_BLOCKS + 4):
            for p in self.RCNN_base[i].parameters(): p.requires_grad=False

        def batch_norm_grad_turn_off(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                for p in module.parameters(): p.requires_grad = False

        self.RCNN_base.apply(batch_norm_grad_turn_off)
        self.RCNN_top.apply(batch_norm_grad_turn_off)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        super().train(mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RCNN_base.eval()
            for i in range(4 + cfg.RESNET.NUM_FREEZE_BLOCKS, 7):
                self.RCNN_base[i].train()

            def batch_norm_eval_turn_on(module):
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()

            self.RCNN_base.apply(batch_norm_eval_turn_on)
            self.RCNN_top.apply(batch_norm_eval_turn_on)

    def _feed_pooled_feature_to_top(self, pooled_feature):
        return self.RCNN_top(pooled_feature).mean(3).mean(2)