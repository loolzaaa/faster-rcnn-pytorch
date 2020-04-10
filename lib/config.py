import os
import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.GENERAL = edict()
cfg.GENERAL.MIN_IMG_RATIO = 0.5
cfg.GENERAL.MAX_IMG_RATIO = 2.0
cfg.GENERAL.PIXEL_MEANS = [102.9801, 115.9465, 122.7717]
cfg.GENERAL.POOLING_SIZE = 7

cfg.TRAIN = edict()
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.LEARNING_RATE = 0.001
cfg.TRAIN.LR_DECAY_STEP = 5
cfg.TRAIN.LR_DECAY_GAMMA = 0.1
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.DOUBLE_BIAS = True
cfg.TRAIN.BIAS_DECAY = False
cfg.TRAIN.WEIGHT_DECAY = 0.0005

cfg.TRAIN.RPN_PRE_NMS_TOP = 12000
cfg.TRAIN.RPN_POST_NMS_TOP = 2000
cfg.TRAIN.RPN_NMS_THRESHOLD = 0.7
cfg.TRAIN.RPN_CLOBBER_POSITIVES = False # If an anchor statisfied by positive and negative conditions set to negative
cfg.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
cfg.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
cfg.TRAIN.RPN_MAX_LABELS = 256
cfg.TRAIN.RPN_FG_LABELS_FRACTION = 0.5

cfg.TRAIN.USE_FLIPPED = True
cfg.TRAIN.PROPOSAL_PER_IMG = 256
cfg.TRAIN.FG_PROPOSAL_FRACTION = 0.25
cfg.TRAIN.FG_THRESHOLD = 0.5
cfg.TRAIN.BG_THRESHOLD_HI = 0.5
cfg.TRAIN.BG_THRESHOLD_LO = 0.0
cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
cfg.TRAIN.BBOX_NORMALIZE_MEANS = (0., 0., 0., 0.)
cfg.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

cfg.RPN = edict()
cfg.RPN.ANCHOR_SCALES = [8, 16, 32]
cfg.RPN.ANCHOR_RATIOS = [0.5, 1, 2]
cfg.RPN.FEATURE_STRIDE = 16

cfg.TEST = edict()
cfg.TEST.RPN_PRE_NMS_TOP = 6000
cfg.TEST.RPN_POST_NMS_TOP = 300
cfg.TEST.RPN_NMS_THRESHOLD = 0.7
cfg.TEST.BBOX_REG = True
cfg.TEST.NMS = 0.3