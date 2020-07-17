# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import cv2
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf

import sys
sys.path.append("../")

from help_utils import tools
from libs.configs import cfgs
from libs.val_libs import voc_eval
from libs.box_utils import draw_box_in_img
from libs.networks import build_whole_network_rel_eval
from data.io.read_tfrecord_batch_rel import next_batch
from data.io.image_preprocess import short_side_resize_for_inference_data

def eval_with_tfrecord(det_net):
    ''' 1. read tfrecord '''
    img_batch, gt_boxes_batch, R_matrix_batch, num_objects_batch, img_h_batch, img_w_batch  = next_batch(1,is_training=False)
    num_objects = num_objects_batch[0]
    img_batch=tf.reshape(img_batch[0,:img_h_batch[0],:img_w_batch[0],:],(1,img_h_batch[0],img_w_batch[0],3))
    gt_boxes = tf.cast(tf.reshape(gt_boxes_batch, [-1, 4]), tf.float32)
    R_matrix = tf.cast(tf.reshape(R_matrix_batch, [num_objects,num_objects]),tf.int32)

    ''' 2. construct network '''
    rel_labels,rel_preds,rel_probs = det_net.build_whole_rel_network(input_img_batch=img_batch, R_matrix=R_matrix, gt_boxes=gt_boxes)

    ''' 3. evaluate '''
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    restorer, rel_restore_ckpt, det_restorer, det_restore_ckpt = det_net.get_restorer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    correct_50_count, all_50_count=0,0
    correct_100_count, all_100_count=0,0

    with tf.Session(config=config) as sess:
        
        sess.run(init_op)

        if not restorer is None:
            restorer.restore(sess, rel_restore_ckpt)
            print('restore relation model.')

        if det_restorer is not None:
            det_restorer.restore(sess,det_restore_ckpt)
            print('restore detection model.')

        for idx in range(cfgs.REL_TEST_COUNT):
            
            start = time.time()
            rel_labels_, rel_preds_, rel_probs_ = sess.run([rel_labels,rel_preds,rel_probs])
            end = time.time()

            rel_preds_[rel_probs_<=cfgs.REL_FILTERED_SCORE]=0

            if rel_preds_.shape[0]>100:
                indices = rel_probs_.argsort()[::-1][:100]
                top_100_labels, top_100_preds = rel_labels_[indices], rel_preds_[indices]
            else:
                top_100_labels, top_100_preds = rel_labels_, rel_preds_
            
            if rel_preds_.shape[0]>50:
                indices = rel_probs_.argsort()[::-1][:50]
                top_50_labels, top_50_preds = rel_labels_[indices], rel_preds_[indices]
            else:
                top_50_labels, top_50_preds = rel_labels_, rel_preds_

            all_100_count+=top_100_preds.shape[0]
            all_50_count+=top_50_preds.shape[0]

            correct_100_count += np.sum( top_100_labels == top_100_preds )
            correct_50_count  += np.sum( top_50_labels  == top_50_preds )        
            
            tools.view_bar('{} image cost {}s'.format(idx, (end - start)), idx + 1, cfgs.REL_TEST_COUNT)
        print('\n','-'*80)
        print('R@50:', 1.0*correct_50_count/all_50_count, 'R@100:', 1.0*correct_100_count/all_100_count)

def eval():
    retinanet = build_whole_network_rel_eval.RelDetectionNetwork(base_network_name=cfgs.NET_NAME, mode='predcls')
    eval_with_tfrecord(det_net=retinanet)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    eval()
















