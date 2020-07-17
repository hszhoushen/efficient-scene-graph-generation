# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import tensorflow as tf

import sys
sys.path.append('../../')

from libs.configs import cfgs
from data.io import image_preprocess_batch

def parse_record(raw_record):
    """Parse data from a tf record."""
    keys_to_features = {
        'img_name': tf.FixedLenFeature([], tf.string),
        'h': tf.FixedLenFeature([], tf.int64),
        'w': tf.FixedLenFeature([], tf.int64),
        'img': tf.FixedLenFeature([], tf.string),
        'r_matrix': tf.FixedLenFeature([], tf.string),
        'gt_boxes': tf.FixedLenFeature([], tf.string),
    }
    # 读入一个样例
    parsed = tf.parse_single_example(raw_record, keys_to_features)
    
    img_name  = parsed['img_name']
    img_height = tf.cast(parsed['h'], tf.int32)
    img_width = tf.cast(parsed['w'], tf.int32)

    img = tf.decode_raw(parsed['img'], tf.uint8)
    img = tf.reshape(img, shape=[img_height, img_width, 3])

    gt_boxes = tf.decode_raw(parsed['gt_boxes'], tf.int32)
    gt_boxes = tf.reshape(gt_boxes, [-1, 5])
    
    num_obs = tf.cast(tf.shape(gt_boxes)[0],tf.int32)
    
    R_matrix = tf.decode_raw(parsed['r_matrix'], tf.int32)
    R_matrix = tf.reshape(R_matrix,[num_obs,num_obs])

    img, gt_boxes, h, w = preprocess_img(img, gt_boxes)#,is_training)

    return img, gt_boxes, R_matrix, num_obs, h, w

def preprocess_img(img, gtbox, is_training=True):
    img = tf.cast(img, tf.float32)
    
    img, gtboxes_and_label, h, w= image_preprocess_batch.short_side_resize(img_tensor=img,
                                                                gtboxes_and_label=gtbox,
                                                                target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                                length_limitation=cfgs.IMG_MAX_LENGTH)
    if is_training:
        img, gtboxes_and_label = image_preprocess_batch.random_flip_left_right(img_tensor=img,
                                                                        gtboxes_and_label=gtboxes_and_label)
    # gtboxes_and_label=gtbox
    if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
        img =(img / 255 - tf.constant([[cfgs.PIXEL_MEAN_]]))/tf.constant([[cfgs.PIXEL_STD]])
    else:
        img = img - tf.constant([[cfgs.PIXEL_MEAN]])  # sub pixel mean at last

    return img, gtboxes_and_label, h, w


def next_batch(batch_size,is_training=True, obj_cls=False):
    ''' 
    img_tensor:[h, w, c], 
    R_matrix:[n,n]
    gt_boxes:[n,4]
    '''
    #assert batch_size == 1, "we only support batch_size is 1.We may support large batch_size in the future"

    if is_training:
        name='train_rel.tfrecord'
    else:
        name='test_rel.tfrecord'

    pattern = os.path.join('/data/scene/', name)
    # pattern = os.path.join('../VGDataset/', name)
    # pattern = os.path.join('../data/VGDataset/', name)
    print('tfrecord path is -->', os.path.abspath(pattern))

    dataset = tf.data.Dataset.from_tensor_slices([pattern])
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    dataset = dataset.map(parse_record)
    dataset = dataset.prefetch(buffer_size=10 * batch_size)
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=100 * batch_size)

    dataset=dataset.padded_batch(batch_size,padded_shapes=([None,None,None],[None,None],[None,None],[],[],[]))

    if is_training:
        dataset = dataset.repeat()

    iterator = dataset.make_one_shot_iterator()
    img, gt_boxes, R_matrix, num_obs, h, w = iterator.get_next()

    if obj_cls:
        return img, gt_boxes, R_matrix, num_obs, h, w
    else:
        return img, gt_boxes[:,:,:4], R_matrix, num_obs, h, w

if __name__ == '__main__':
    
    import cv2
    import time

    img_batch, gt_boxes_batch, R_matrix_batch, num_objects_batch, img_h_batch, img_w_batch =next_batch(2,is_training=False)

    inputs_list = []
    for i in range(2):
        num_objects = num_objects_batch[i]
        img_h = img_h_batch[i]
        img_w = img_w_batch[i]
        img=tf.reshape(img_batch[i,:img_h,:img_w,:],(img_h,img_w,3))
        gtboxes = tf.cast(tf.reshape(gt_boxes_batch[i,:num_objects,:], [-1, 4]), tf.float32)
        R_matrix = tf.cast(tf.reshape(R_matrix_batch[i,:num_objects,:num_objects], [num_objects,num_objects]),tf.float32)
        inputs_list.append([img, gtboxes, R_matrix])
    print('-'*40)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        count=0
        while True:
            for idx in range(2):
                start = time.time()
                i,gt,r=sess.run(inputs_list[idx])
                end = time.time()
                # print(i.shape)
                print(gt.shape)
                # print(r.shape)

                # print(r)
                # print(np.sum(r>0))

                # i = (i*np.array(cfgs.PIXEL_STD)+ np.array(cfgs.PIXEL_MEAN_)) * 255
                # i = np.array(i * 255 / np.max(i), dtype=np.uint8)

                # rel_img=i[:,:,::-1].copy()
                # for i in range(r.shape[0]):
                #     for j in range(r.shape[1]):
                #         if r[i,j]>0:
                #             sx1, sy1, sx2, sy2 = gt[i]
                #             cv2.rectangle(rel_img,(sx1, sy1),(sx2, sy2),(255,0,0),2,0)
                #             ox1, oy1, ox2, oy2 = gt[j]
                #             cv2.rectangle(rel_img,(ox1, oy1),(ox2, oy2),(0,255,0),2,0)
                            
                #             x1=int(min(ox1,sx1)-3)
                #             y1=int(min(oy1,sy1)-3)
                #             x2=int(max(ox2,sx2)+3)
                #             y2=int(max(oy2,sy2)+3)
                #             cv2.rectangle(rel_img,(x1, y1),(x2,y2),(0,0,255),2,0)

                # cv2.imshow('rel_img',rel_img)
                # cv2.waitKey(1)

            count+=1
            print(count)
        

        coord.request_stop()
        coord.join(threads)
