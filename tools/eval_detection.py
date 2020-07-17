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
from libs.networks import build_whole_network
from data.io.read_tfrecord_batch_detection import next_batch
from data.io.image_preprocess import short_side_resize_for_inference_data

def eval_with_tfrecord(det_net, draw_imgs=False):
    pred_boxes,gt_boxes=[],[]
    ''' 1. read tfrecord '''
    # [1,h,w,c],[1,b,5]
    img_batch, gtboxes_and_label_batch, num_objects_batch, img_h_batch, img_w_batch = next_batch(1,is_training=False)
    img_batch=tf.reshape(img_batch[0,:img_h_batch[0],:img_w_batch[0],:],(1,img_h_batch[0],img_w_batch[0],3))
    gtboxes_and_label=tf.reshape(gtboxes_and_label_batch,[-1,5])
    
    ''' 2. construct network '''
    detection_boxes, detection_scores, detection_category = det_net.build_whole_detection_network(input_img_batch=img_batch, gtboxes_batch=None)

    ''' 3. evaluate '''
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    restorer, restore_ckpt = det_net.get_restorer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        for idx in range(cfgs.DETECTION_TEST_COUNT):
            start = time.time()
            img, gtboxes_label, detected_boxes, detected_scores, detected_categories = \
                sess.run([img_batch, gtboxes_and_label, detection_boxes, detection_scores, detection_category])
            end = time.time()
            
            if draw_imgs:
                show_indices = detected_scores >= cfgs.VIS_SCORE
                show_scores = detected_scores[show_indices]
                show_boxes = detected_boxes[show_indices]
                show_categories = detected_categories[show_indices]

                draw_img = np.squeeze(img, 0)
                if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
                    draw_img = (draw_img * np.array(cfgs.PIXEL_STD) + np.array(cfgs.PIXEL_MEAN_)) * 255
                else:
                    draw_img = draw_img + np.array(cfgs.PIXEL_MEAN)
                final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(draw_img,
                                                                                    boxes=show_boxes,
                                                                                    labels=show_categories,
                                                                                    scores=show_scores,
                                                                                    in_graph=False)

                # gt_detections = draw_box_in_img.draw_boxes_with_label_and_scores(draw_img,
                #                                                                 boxes=gtboxes_label[:,:-1],
                #                                                                 labels=gtboxes_label[:,-1],
                #                                                                 scores=np.ones_like(gtboxes_label[:,-1]),
                #                                                                 in_graph=False)

                if not os.path.exists(os.path.join(cfgs.TEST_SAVE_PATH, cfgs.DETECTION_VERSION)):
                    os.makedirs(os.path.join(cfgs.TEST_SAVE_PATH, cfgs.DETECTION_VERSION))
                cv2.imwrite(os.path.join(cfgs.TEST_SAVE_PATH, cfgs.DETECTION_VERSION, str(idx) + '.jpg'), final_detections[:, :, ::-1])
                # cv2.imwrite(os.path.join(cfgs.TEST_SAVE_PATH, cfgs.DETECTION_VERSION, str(idx) + '_g.jpg'), gt_detections[:, :, ::-1])

            xmin, ymin, xmax, ymax = detected_boxes[:, 0], detected_boxes[:, 1], detected_boxes[:, 2], detected_boxes[:, 3]

            boxes = np.transpose(np.stack([xmin, ymin, xmax, ymax]))
            dets = np.hstack((detected_categories.reshape(-1, 1), detected_scores.reshape(-1, 1), boxes))
            pred_boxes.append(dets)
            gt_boxes.append(gtboxes_label)

            tools.view_bar('{} image cost {}s'.format(idx, (end - start)), idx + 1, cfgs.DETECTION_TEST_COUNT)
        
        return pred_boxes,gt_boxes

def eval(showbox):
    retinanet = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME, is_training=False)
    pred_boxes, gt_boxes = eval_with_tfrecord(det_net=retinanet, draw_imgs=showbox)

    voc_eval.voc_evaluate_detections(pres_boxes=pred_boxes,
                                     gt_boxes=gt_boxes,
                                     test_imgid_list=[i for i in range(len(pred_boxes))])

def parse_args():
    ''' single gpu '''
    parser = argparse.ArgumentParser('evaluate the result with Pascal2007 stdand')
    parser.add_argument('--gpu',      dest='gpu',      type=str, help='gpu id', default='0')
    parser.add_argument('--showbox',  action='store_true',       help='whether show detecion results when evaluation')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(20*"--")
    print(args)
    print(20*"--")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    eval(showbox=args.showbox)
















