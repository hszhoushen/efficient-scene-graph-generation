# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import os
import math
import tensorflow as tf

# ------------------------------------------------
REL_VERSION = 'RN_SG_20200523'
#DETECTION_VERSION = 'RN_DE_20200518'
DETECTION_VERSION = 'RN_DE_20200602'
NET_NAME = 'resnet50_v1d'  # 'MobilenetV2'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,1,2,3"
NUM_GPU = len(GPU_GROUP.strip().split(','))

if NET_NAME.startswith("resnet"):
    weights_name = NET_NAME
elif NET_NAME.startswith("MobilenetV2"):
    weights_name = "mobilenet/mobilenet_v2_1.0_224"
else:
    raise Exception('net name must in [resnet_v1_101, resnet_v1_50, MobilenetV2]')

SUMMARY_PATH = ROOT_PATH + '/output/summary'
PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False
FIXED_BLOCKS = 1  # allow 0~3
FREEZE_BLOCKS = [True, False, False, False, False]  # for gluoncv backbone
USE_07_METRIC = True

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
REL_MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip

SHOW_TRAIN_INFO_INTE = 20
SMRY_ITER = 200
SAVE_WEIGHTS_INTE = 80000
REL_SAVE_WEIGHTS_INTE = 60000

BATCH_SIZE =1
REL_BATCH_SIZE = 128
REL_FRACTION = 0.5
EPSILON = 1e-5
MOMENTUM = 0.9

LR = 5e-4 * NUM_GPU * BATCH_SIZE
DECAY_STEP = [SAVE_WEIGHTS_INTE*12, SAVE_WEIGHTS_INTE*16, SAVE_WEIGHTS_INTE*20]
MAX_ITERATION = SAVE_WEIGHTS_INTE*20
WARM_SETP = int(1*SAVE_WEIGHTS_INTE)

REL_LR = 1e-3 
REL_DECAY_STEP = [REL_SAVE_WEIGHTS_INTE*6, REL_SAVE_WEIGHTS_INTE*12, REL_SAVE_WEIGHTS_INTE*18]
REL_MAX_ITERATION = REL_SAVE_WEIGHTS_INTE*18
REL_WARM_SETP = int(1*REL_SAVE_WEIGHTS_INTE)

# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'vg'  
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 600
IMG_MAX_LENGTH = 1000
REL_CLASS_NUM = 50
OBJ_CLASS_NUM = 150

# --------------------------------------------- Network_config
SUBNETS_WEIGHTS_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
SUBNETS_BIAS_INITIALIZER = tf.constant_initializer(value=0.0)
PROBABILITY = 0.01
FINAL_CONV_BIAS_INITIALIZER = tf.constant_initializer(value=-math.log((1.0 - PROBABILITY) / PROBABILITY))
WEIGHT_DECAY = 1e-4

# ---------------------------------------------Anchor config
LEVEL = ['P3', 'P4', 'P5', 'P6', 'P7']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE = [8, 16, 32, 64, 128]
ANCHOR_SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
ANCHOR_RATIOS = [0.5, 1.0, 2.0]
ANCHOR_SCALE_FACTORS = None
USE_CENTER_OFFSET = True

# --------------------------------------------RPN config
SHARE_NET = True
USE_P5 = True
IOU_POSITIVE_THRESHOLD = 0.5
IOU_NEGATIVE_THRESHOLD = 0.4

ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
CHANNELS_FEATURE = 256

NMS = True
NMS_IOU_THRESHOLD = 0.5
MAXIMUM_DETECTIONS = 100
FILTERED_SCORE = 0.25
VIS_SCORE = 0.5

REL_VIS_SCORE = 0.5
REL_FILTERED_SCORE = 0.5

# --------------------------------------------Detection evaluate
DETECTION_TRAIN_COUNT=84111
DETECTION_TEST_COUNT=21032

REL_TRAIN_COUNT=59167
REL_TEST_COUNT=14649

TEST_SAVE_PATH = ROOT_PATH + '/tools/demos/detection_img_result/'
EVALUATE_DIR = ROOT_PATH + '/tools/demos/detection_eval_result/'
