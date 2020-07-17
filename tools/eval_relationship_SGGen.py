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

def _gt_triplet(gt_boxes,R_matrix):

    triplets, triplet_boxes = [], []
    for i in range(R_matrix.shape[0]):
        for j in range(R_matrix.shape[1]):
            if R_matrix[i,j]<=0:
                continue
            t=[gt_boxes[i,4],R_matrix[i,j],gt_boxes[j,4]]
            sx1,sy1,sx2,sy2=gt_boxes[i,:4]
            ox1,oy1,ox2,oy2=gt_boxes[j,:4]
            b=[sx1,sy1,sx2,sy2,ox1,oy1,ox2,oy2]
            triplets.append(t)
            triplet_boxes.append(b)

    triplets=np.array(triplets,dtype=np.int32)
    triplet_boxes=np.array(triplet_boxes,dtype=np.int32)

    return triplets, triplet_boxes

def _pred_triplet(rel_scores, rel_categorys, sub_boxes ,obj_boxes, sub_categorys, obj_categorys):

    triplets, triplet_boxes, triplet_scores = [], [], []
    for idx in range(rel_scores.shape[0]):
        t=[sub_categorys[idx], rel_categorys[idx], obj_categorys[idx]]
        sx1,sy1,sx2,sy2 = sub_boxes[idx]
        ox1,oy1,ox2,oy2 = obj_boxes[idx]
        b=[sx1, sy1, sx2, sy2, ox1, oy1, ox2, oy2]
        triplets.append(t)
        triplet_boxes.append(b)
        triplet_scores.append(rel_scores[idx])

    triplets=np.array(triplets,dtype=np.int32)
    triplet_boxes=np.array(triplet_boxes,dtype=np.int32)
    triplet_scores=np.array(triplet_scores,dtype=np.float32)
    return triplets, triplet_boxes, triplet_scores

def _iou(gt_box, pred_boxes):
    # computer Intersection-over-Union between two sets of boxes
    ixmin = np.maximum(gt_box[0], pred_boxes[:,0])
    iymin = np.maximum(gt_box[1], pred_boxes[:,1])
    ixmax = np.minimum(gt_box[2], pred_boxes[:,2])
    iymax = np.minimum(gt_box[3], pred_boxes[:,3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) +
            (pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) *
            (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps

def _relation_recall(gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thresh=0.5):
    # compute the R@K metric for a set of predicted triplets
    num_gt = gt_triplets.shape[0]
    num_correct_pred_gt = 0

    for gt, gt_box in zip(gt_triplets, gt_boxes):
        keep = np.zeros(pred_triplets.shape[0]).astype(bool)
        for i, pred in enumerate(pred_triplets):
            if gt[0] == pred[0] and gt[1] == pred[1] and gt[2] == pred[2]:
                keep[i] = True
        if not np.any(keep):
            continue
        boxes = pred_boxes[keep,:]
        sub_iou = _iou(gt_box[:4], boxes[:,:4])
        obj_iou = _iou(gt_box[4:], boxes[:,4:])
        inds = np.intersect1d(np.where(sub_iou >= iou_thresh)[0],
                              np.where(obj_iou >= iou_thresh)[0])
        if inds.size > 0:
            num_correct_pred_gt += 1
    return float(num_correct_pred_gt), float(num_gt)

def eval_with_tfrecord(det_net):
    ''' 1. read tfrecord '''
    # gt_boxes:[n,5] :(x1,y1,x2,y2,c)
    img_batch, gt_boxes_batch, R_matrix_batch, num_objects_batch, img_h_batch, img_w_batch  = next_batch(1,is_training=False,obj_cls=True)
    num_objects = num_objects_batch[0]
    img_batch=tf.reshape(img_batch[0,:img_h_batch[0],:img_w_batch[0],:],(1,img_h_batch[0],img_w_batch[0],3))
    gt_boxes = tf.cast(tf.reshape(gt_boxes_batch, [-1, 5]), tf.float32)
    R_matrix = tf.cast(tf.reshape(R_matrix_batch, [num_objects,num_objects]),tf.int32)

    ''' 2. construct network '''
    rel_prob,rel_preds,sub_boxes,obj_boxes,sub_categorys,obj_categorys = \
        det_net.build_whole_rel_network(input_img_batch=img_batch, R_matrix=None, gt_boxes=None)

    ''' 3. evaluate '''
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    restorer, rel_restore_ckpt, det_restorer, det_restore_ckpt = det_net.get_restorer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    correct_50_count, all_50_count=0,0
    correct_100_count, all_100_count=0,0
    all_time, all_count=0,0
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
            gt_boxes_, R_matrix_, rel_probs_, rel_preds_, sub_boxes_ ,obj_boxes_ ,sub_categorys_ ,obj_categorys_ = \
                sess.run([gt_boxes,R_matrix,rel_prob,rel_preds,sub_boxes,obj_boxes,sub_categorys,obj_categorys])
            end = time.time()
            all_time+=(end-start)
            all_count+=1
	
            gt_triplets, gt_triplets_boxes = _gt_triplet(gt_boxes_,R_matrix_)
            
            ''' Recall@100 '''
            if rel_preds_.shape[0]>100:
                indices = rel_probs_.argsort()[::-1][:100]
                rel_probs_, rel_preds_, sub_boxes_ ,obj_boxes_, sub_categorys_, obj_categorys_ = \
                    rel_probs_[indices], rel_preds_[indices], sub_boxes_[indices], obj_boxes_[indices], sub_categorys_[indices], obj_categorys_[indices]

            # rel_preds_[rel_probs_<=cfgs.REL_FILTERED_SCORE]=0
            indices=np.where(rel_preds_>0)

            rel_probs_, rel_preds_, sub_boxes_ ,obj_boxes_, sub_categorys_, obj_categorys_ = \
                rel_probs_[indices], rel_preds_[indices], sub_boxes_[indices], obj_boxes_[indices], sub_categorys_[indices], obj_categorys_[indices]
            
            pred_triplets, pred_triplet_boxes, pred_triplet_scores = _pred_triplet(rel_probs_, rel_preds_, sub_boxes_ ,obj_boxes_, sub_categorys_, obj_categorys_)

            pcn,gn=_relation_recall(gt_triplets,pred_triplets,gt_triplets_boxes,pred_triplet_boxes)
            correct_100_count+=pcn
            all_100_count+=gn

            ''' Recall@50 '''
            if rel_preds_.shape[0]>50:
                indices = rel_probs_.argsort()[::-1][:50]
                rel_probs_, rel_preds_, sub_boxes_ ,obj_boxes_, sub_categorys_, obj_categorys_ = \
                    rel_probs_[indices], rel_preds_[indices], sub_boxes_[indices], obj_boxes_[indices], sub_categorys_[indices], obj_categorys_[indices]

            # rel_preds_[rel_probs_<=cfgs.REL_FILTERED_SCORE]=0
            indices=np.where(rel_preds_>0)

            rel_probs_, rel_preds_, sub_boxes_ ,obj_boxes_, sub_categorys_, obj_categorys_ = \
                rel_probs_[indices], rel_preds_[indices], sub_boxes_[indices], obj_boxes_[indices], sub_categorys_[indices], obj_categorys_[indices]
            
            pred_triplets, pred_triplet_boxes, pred_triplet_scores = _pred_triplet(rel_probs_, rel_preds_, sub_boxes_ ,obj_boxes_, sub_categorys_, obj_categorys_)

            pcn,gn=_relation_recall(gt_triplets,pred_triplets,gt_triplets_boxes,pred_triplet_boxes)
            correct_50_count+=pcn
            all_50_count+=gn 
            
            tools.view_bar('{} image cost {}s'.format(idx, (end - start)), idx + 1, cfgs.REL_TEST_COUNT)
        print('R@50:', 1.0*correct_50_count/all_50_count, 'R@100:', 1.0*correct_100_count/all_100_count,' average time:',1.0*all_time/all_count)
def eval():
    retinanet = build_whole_network_rel_eval.RelDetectionNetwork(base_network_name=cfgs.NET_NAME, mode='sggen')
    eval_with_tfrecord(det_net=retinanet)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    eval()
