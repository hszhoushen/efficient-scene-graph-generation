# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import sys
sys.path.append("../../")

from libs.configs import cfgs

if cfgs.OBJ_CLASS_NUM>=151:
        obj_class_names = [
                'back_ground','roof','kite','pant','bowl','laptop','paper','shoe',
                'railing','chair','windshield','ear','tire','cup','bench','tail',
                'bike','board','orange','hat','finger','plate','woman','handle',
                'branch','food','elephant','bear','wave','tile','giraffe','desk',
                'lady','towel','glove','bag','nose','rock','tower','motorcycle',
                'sneaker','fence','people','house','sign','hair','street','zebra',
                'racket','logo','girl','arm','wire','leaf','clock','hill','bird',
                'umbrella','leg','screen','men','sink','trunk','post','sidewalk',
                'box','boy','cow','skateboard','plane','stand','pillow','toilet',
                'pot','number','pole','table','boat','sheep','horse','eye','sock',
                'window','vehicle','curtain','man','banana','fork','head','door',
                'shelf','cabinet','glass','flag','train','child','seat','neck',
                'room','player','ski','cap','tree','bed','cat','light','skier',
                'engine','drawer','guy','airplane','car','mountain','shirt','paw',
                'boot','snow','lamp','book','flower','animal','bus','vegetable',
                'tie','beach','pizza','wheel','plant','helmet','track','hand',
                'fruit','mouth','letter','vase','kid','building','short','surfboard'
                ,'phone','coat','counter','dog','face','jacket','person','truck'
                ,'bottle','basket','jean','wing','other']
else:
        obj_class_names = [
                'back_ground','roof','kite','pant','bowl','laptop','paper','shoe',
                'railing','chair','windshield','ear','tire','cup','bench','tail',
                'bike','board','orange','hat','finger','plate','woman','handle',
                'branch','food','elephant','bear','wave','tile','giraffe','desk',
                'lady','towel','glove','bag','nose','rock','tower','motorcycle',
                'sneaker','fence','people','house','sign','hair','street','zebra',
                'racket','logo','girl','arm','wire','leaf','clock','hill','bird',
                'umbrella','leg','screen','men','sink','trunk','post','sidewalk',
                'box','boy','cow','skateboard','plane','stand','pillow','toilet',
                'pot','number','pole','table','boat','sheep','horse','eye','sock',
                'window','vehicle','curtain','man','banana','fork','head','door',
                'shelf','cabinet','glass','flag','train','child','seat','neck',
                'room','player','ski','cap','tree','bed','cat','light','skier',
                'engine','drawer','guy','airplane','car','mountain','shirt','paw',
                'boot','snow','lamp','book','flower','animal','bus','vegetable',
                'tie','beach','pizza','wheel','plant','helmet','track','hand',
                'fruit','mouth','letter','vase','kid','building','short','surfboard'
                ,'phone','coat','counter','dog','face','jacket','person','truck'
                ,'bottle','basket','jean','wing']
        
        # obj_class_names=['back_ground', 'person', 'bicycle', 'car', 'motorcycle',
        # 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        # 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        # 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
        # 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        # 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        # 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        # 'skateboard', 'surfboard', 'tennis racket', 'bottle',
        # 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        # 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        # 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        # 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        # 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        # 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        # 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        # 'hair drier', 'toothbrush']

obj_classes_ID = dict(zip(obj_class_names[1:],range(1,len(obj_class_names)+1)))
# print(len(obj_class_names))
# print(obj_classes_ID)
if cfgs.REL_CLASS_NUM<=50:
        rel_class_names = [
                'back_ground','and','says','belonging to','over','parked on','growing on',
                'standing on','made of','attached to','at','in','hanging from','in front of',
                'from','for','lying on','to','behind','flying in','looking at','on back of',
                'holding','under','laying on','riding','has','across','wearing','walking on',
                'eating','wears','watching','walking in','sitting on','between','covered in',
                'carrying','using','along','on','with','above','part of','covering','of',
                'against','playing','near','painted on','mounted on']
else:
        rel_class_names = [
                'back_ground','and','says','belonging to','over','parked on','growing on',
                'standing on','made of','attached to','at','in','hanging from','in front of',
                'from','for','lying on','to','behind','flying in','looking at','on back of',
                'holding','under','laying on','riding','has','across','wearing','walking on',
                'eating','wears','watching','walking in','sitting on','between','covered in',
                'carrying','using','along','on','with','above','part of','covering','of',
                'against','playing','near','painted on','mounted on','other']

rel_classes_ID = dict(zip(rel_class_names[1:],range(1,len(rel_class_names)+1)))
# print(len(rel_class_names))
# print(rel_classes_ID)
