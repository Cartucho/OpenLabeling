from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import sys
import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from os.path import realpath, dirname, join, exists
# set device, depending on whether cuda is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class SiamMask(object):


    def __init__(self,classid=0,anchorid=0,init_frame=None,init_bbox=None):
        # load config
        cfg_path = '../object_detection/pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml'
        snapshot = '../object_detection/pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth'
        cfg.merge_from_file(cfg_path)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        # cfg.CUDA = False
        device = torch.device('cuda' if cfg.CUDA else 'cpu')
        # device='cpu'

        # create model
        model = ModelBuilder()

        # load model
        model.load_state_dict(torch.load(snapshot,
            map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(device)

        # build tracker
        tracker = build_tracker(model)
        
        self.tracker = tracker
        self.classId = classid
        self.anchorId = anchorid
        self.init_bbox = init_bbox
        self.init_frame = init_frame
            

    def init(self, init_frame, initial_bbox):
        """
        Initialize DaSiamRPN tracker with inital frame and bounding box.
        """
        with torch.no_grad():
            self.tracker.init(init_frame,initial_bbox)
        # target_pos, target_sz = self.bbox_to_pos(initial_bbox)
        # self.state = SiamRPN_init(
        #     init_frame, target_pos, target_sz, self.net)

    def init_reinit(self):
        self.init(self.init_frame,self.init_bbox)

    def update(self, next_image):
        """
        Update bounding box position and size on next_image. Returns True
        beacuse tracking is terminated based on number of frames predicted
        in OpenLabeling, not based on feedback from tracking algorithm (unlike
        the opencv tracking algorithms).
        """
        with torch.no_grad():
            # self.state = SiamRPN_track(self.state, next_image)
            # target_pos = self.state["target_pos"]
            # target_sz  = self.state["target_sz"]
            # bbox = self.pos_to_bbox(target_pos, target_sz)
            outputs = self.tracker.track(next_image)
            # if 'polygon' in outputs:
            #     polygon = np.array(outputs['polygon']).astype(np.int32)
            #     cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
            #                     True, (0, 255, 0), 3)
            #     mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
            #     mask = mask.astype(np.uint8)
            #     mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
            #     frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            # else:
            bbox = list(map(int, outputs['bbox']))
            # cv2.rectangle(frame, (bbox[0], bbox[1]),
            #                 (bbox[0]+bbox[2], bbox[1]+bbox[3]),
            #                 (0, 255, 0), 3)
            score = outputs['best_score']
            if score > 0.7:
                return True, bbox
            else:
                return False, bbox

            # return True, bbox

    def bbox_to_pos(self, initial_bbox):
        """
        Convert bounding box format from a tuple format containing
        xmin, ymin, width, and height to a tuple of two arrays which contain
        the x and y coordinates of the center of the box and its width and
        height respectively.
        """
        xmin, ymin, w, h = initial_bbox
        cx = int(xmin + w/2)
        cy = int(ymin + h/2)
        target_pos = np.array([cx, cy])
        target_sz  = np.array([w, h])

        return target_pos, target_sz

    def pos_to_bbox(self, target_pos, target_sz):
        """
        Invert the bounding box format produced in the above conversion
        function.
        """
        w = target_sz[0]
        h = target_sz[1]
        xmin = int(target_pos[0] - w/2)
        ymin = int(target_pos[1] - h/2)

        return xmin, ymin, w, h
