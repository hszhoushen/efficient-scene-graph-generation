# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
sys.path.append("../")

import os
import cv2  
import time
import argparse
import numpy as np
import tensorflow as tf

from help_utils import tools
from libs.configs import cfgs
from libs.box_utils import draw_box_in_img
from libs.label_name_dict.label_dict import *
from libs.networks import build_whole_network
from data.io.image_preprocess import short_side_resize_for_inference_data

def detect(det_net, imgname_list, save_path=None, label_path=None, top_k=10):
    ''' 1. preprocess img '''
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not BGR
    img_batch = tf.cast(img_plac, tf.float32)
    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch, target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN, length_limitation=cfgs.IMG_MAX_LENGTH)
                                                     
    if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
        img_batch = (img_batch / 255 - tf.constant(cfgs.PIXEL_MEAN_)) / tf.constant(cfgs.PIXEL_STD)
    else:
        img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = tf.expand_dims(img_batch, axis=0)

    ''' 2. construct network '''
    obj_boxes, obj_scores, obj_category = det_net.build_whole_detection_network(input_img_batch=img_batch, gtboxes_batch=None)

    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    restorer, restore_ckpt = det_net.get_restorer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model from ',restore_ckpt)

        for i, a_img_name in enumerate(imgname_list):
            raw_img = cv2.imread(a_img_name)

            start = time.time()
            resized_img, show_boxes, show_scores, show_categories = sess.run([img_batch, obj_boxes, obj_scores, obj_category], feed_dict={img_plac: raw_img[:, :, ::-1]})
            end = time.time()
            print("{} cost time : {} ".format(a_img_name, (end - start)))

            if show_scores.shape[0]>top_k:
                show_indices = show_scores.argsort()[::-1][:top_k]
                show_boxes, show_scores, show_categories=show_boxes[show_indices], show_scores[show_indices], show_categories[show_indices]

            draw_img = np.squeeze(resized_img,0)
            if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
                draw_img = (draw_img * np.array(cfgs.PIXEL_STD) + np.array(cfgs.PIXEL_MEAN_)) * 255
            else:
                draw_img = draw_img + np.array(cfgs.PIXEL_MEAN)
            final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(draw_img, boxes=show_boxes, labels=show_categories, scores=show_scores, in_graph=False, detection=True)

            cv2.imshow('img',final_detections[:,:,::-1])

            if label_path is not None:
                with open(os.path.join(label_path,a_img_name.split('/')[-1].split('.')[0]+'.txt'),'r') as f:
                    gts=f.readlines()
                boxes,labels=[],[]
                for gt in gts:
                    info=gt.strip().split(',')
                    boxes.append([int(info[0]),int(info[1]),int(info[2]),int(info[3])])
                    boxes.append([int(info[5]),int(info[6]),int(info[7]),int(info[8])])
                    labels.append(int(info[4]))
                    labels.append(int(info[9]))
                boxes=np.array(boxes)*(draw_img.shape[0]/raw_img.shape[0])
                labels=np.array(labels)
                scores=np.ones_like(labels)
                gt_img=draw_box_in_img.draw_boxes_with_label_and_scores(draw_img,boxes,labels,scores,False,True)
                cv2.imshow('gt_img',gt_img[:,:,::-1])

            if save_path:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                idx=a_img_name.split('/')[-1].split('.')[0]
                cv2.imwrite(save_path + '/' + idx +'.jpg',final_detections[:,:,::-1])
                if label_path is not None:
                    cv2.imwrite(label_path + '/' + idx +'.jpg',gt_img[:,:,::-1])

            cv2.waitKey(100)
            tools.view_bar('{} image cost {}s'.format(a_img_name, (end - start)), i + 1, len(imgname_list))


def inference(data_path, save_path, label_path=None, top_k=100):
    test_imgname_list = [os.path.join(data_path, img_name) for img_name in os.listdir(data_path) if img_name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
                                                          
    assert len(test_imgname_list) != 0, 'test_dir has no imgs there. Note that, we only support img format of (.jpg, .png, and .tiff) '

    retinanet = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME, is_training=False)
    
    detect(det_net=retinanet, 
           imgname_list=test_imgname_list,
           save_path=save_path, 
           label_path=label_path,
           top_k=top_k)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='TestImgs...U need provide the test dir')
    parser.add_argument('--data_dir', dest='data_dir', help='data path', default='demos', type=str)
    parser.add_argument('--label_dir', dest='label_dir', help='label path', default=None, type=str)
    parser.add_argument('--save_dir', dest='save_dir', help='demo imgs to save', default=None, type=str)
    parser.add_argument('--top_k', help='top k', default=100, type=int)
    parser.add_argument('--GPU', dest='GPU', help='gpu id ', default='0', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    inference(data_path  = args.data_dir,
              save_path  = args.save_dir,
              label_path = args.label_dir,
              top_k      = args.top_k)















