# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import numpy as np
import xml.etree.ElementTree as ET

from libs.configs import cfgs
from help_utils.tools import mkdir
from libs.label_name_dict.label_dict import obj_class_names

NAME_LABEL_MAP=obj_class_names

def write_voc_results_file(all_boxes, test_imgid_list, det_save_dir):
  '''

  :param all_boxes: is a list. each item reprensent the detections of a img.
  the detections is a array. shape is [-1, 6]. [category, score, xmin, ymin, xmax, ymax]
  Note that: if none detections in this img. that the detetions is : []

  :param test_imgid_list:
  :param det_save_path:
  :return:
  '''
  for cls_id,cls in enumerate(NAME_LABEL_MAP):
    if cls == 'back_ground':
      continue
    print("Writing {} predicted resutls file".format(cls))

    mkdir(det_save_dir)
    det_save_path = os.path.join(det_save_dir, "det_"+cls+".txt")

    with open(det_save_path, 'wt') as f:
      for index, img_name in enumerate(test_imgid_list):
        this_img_detections = all_boxes[index]

        this_cls_detections = this_img_detections[this_img_detections[:, 0]==cls_id]
        if this_cls_detections.shape[0] == 0:
          continue # this cls has none detections in this img
        for a_det in this_cls_detections:
          f.write('{:d} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                  format(img_name, a_det[1],
                         a_det[2], a_det[3],
                         a_det[4], a_det[5]))  # that is [img_name, score, xmin, ymin, xmax, ymax]

def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def voc_eval(detpath, gt_boxes, test_imgid_list, cls_id, ovthresh=0.5, use_07_metric=False):
  # get gtboxes for this class.
  class_recs = {}
  num_pos = 0
  for idx,gt in enumerate(gt_boxes):
    R = [obj for obj in gt if obj[4] == cls_id]
    bbox = np.array([x[:4] for x in R])
    num_pos+=len(R)
    det = [False] * len(R)
    class_recs[idx] = {'bbox': bbox, 'det': det} # det means that gtboxes has already been detected
  
  # 3. read the detection file
  detfile = os.path.join(detpath, "det_"+NAME_LABEL_MAP[cls_id]+".txt")
  with open(detfile, 'r') as f:
    lines = f.readlines()

  # for a line. that is [img_name, confidence, xmin, ymin, xmax, ymax]
  splitlines = [x.strip().split(' ') for x in lines]  # a list that include a list
  image_ids = [int(x[0]) for x in splitlines]  # img_id is img_name
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

  nd = len(image_ids) # num of detections. That, a line is a det_box.
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]  # reorder the img_name

    # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]  # img_id is img_name
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not R['det'][jmax]:
          tp[d] = 1.
          R['det'][jmax] = 1
        else:
          fp[d] = 1.
      else:
        fp[d] = 1.

  # 4. get recall, precison and AP
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(num_pos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric)
  return rec, prec, ap


def do_python_eval(test_imgid_list, gt_boxes):
  AP_list = []

  f=open(os.path.join(cfgs.EVALUATE_DIR, cfgs.DETECTION_VERSION, 'eval_result.txt'),'w')
  for index,cls in enumerate(NAME_LABEL_MAP):
    if cls == 'back_ground':
      continue
    recall, precision, AP = voc_eval(detpath=os.path.join(cfgs.EVALUATE_DIR, cfgs.DETECTION_VERSION),
                                     test_imgid_list=test_imgid_list,
                                     cls_id=index,
                                     gt_boxes=gt_boxes,
                                     use_07_metric=cfgs.USE_07_METRIC)
    if not len(recall):
      s="cls : {} no evaluate examples".format(cls)
    else:
      AP_list += [AP]
      s="cls : {}|| Recall: {} || Precison: {}|| AP: {}".format(cls, recall[-1], precision[-1], AP)
    print(s)
    f.write(s+'\n')

  s="mAP is : {}".format(np.mean(AP_list))
  print(s)
  f.write('\n'+s+'\n')
  f.close()

def voc_evaluate_detections(pres_boxes, gt_boxes, test_imgid_list):
  '''

  :param all_boxes: is a list. each item reprensent the detections of a img.

  The detections is a array. shape is [-1, 6]. [category, score, xmin, ymin, xmax, ymax]
  Note that: if none detections in this img. that the detetions is : []
  :return:
  '''
  write_voc_results_file(pres_boxes, test_imgid_list=test_imgid_list, det_save_dir=os.path.join(cfgs.EVALUATE_DIR, cfgs.DETECTION_VERSION))

  do_python_eval(test_imgid_list, gt_boxes=gt_boxes)







