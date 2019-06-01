"""
Author : Will Stone
Date   : 190407
Desc   : Wrapper class for the DaSiamRPN tracking method. This class has the
         methods required to interface with the tracking class implemented
         in main.py within the OpenLabeling package.
"""
import torch
import numpy as np
import sys
from os.path import realpath, dirname, join, exists
try:
    from DaSiamRPN.code.run_SiamRPN import SiamRPN_init, SiamRPN_track
except ImportError:
    # check if the user has downloaded the submodules
    if not exists(join('DaSiamRPN', 'code', 'net.py')):
        print('Error: DaSiamRPN files not found. Please run the following command:')
        print('\tgit submodule update --init')
        exit()
    else:
        # if python 3
        if sys.version_info >= (3, 0):
            sys.path.append(realpath(join('DaSiamRPN', 'code')))
        else:
            # check if __init__py files exist (otherwise create them)
            path_temp = join('DaSiamRPN', 'code', '__init__.py')
            if not exists(path_temp):
                open(path_temp, 'w').close()
            path_temp = join('DaSiamRPN', '__init__.py')
            if not exists(path_temp):
                open(path_temp, 'w').close()
        # try to import again
        from DaSiamRPN.code.run_SiamRPN import SiamRPN_init, SiamRPN_track
from DaSiamRPN.code.net import SiamRPNvot
from DaSiamRPN.code.utils import get_axis_aligned_bbox, cxy_wh_2_rect

class dasiamrpn(object):
    """
    Wrapper class for incorporating DaSiamRPN into OpenLabeling
    (https://github.com/foolwood/DaSiamRPN,
    https://github.com/Cartucho/OpenLabeling)
    """
    def __init__(self):
        self.net = SiamRPNvot()
        # check if SiamRPNVOT.model was already downloaded (otherwise download it now)
        model_path = join(realpath(dirname(__file__)), 'DaSiamRPN', 'code', 'SiamRPNVOT.model')
        print(model_path)
        if not exists(model_path):
            print('\nError: module not found. Please download the pre-trained model and copy it to the directory \'DaSiamRPN/code/\'\n')
            print('\tdownload link: https://drive.google.com/file/d/1-vNVZxfbIplXHrqMHiJJYWXYWsOIvGsf/view')
            exit()
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval().cuda()

    def init(self, init_frame, initial_bbox):
        """
        Initialize DaSiamRPN tracker with inital frame and bounding box.
        """
        target_pos, target_sz = self.bbox_to_pos(initial_bbox)
        self.state = SiamRPN_init(
            init_frame, target_pos, target_sz, self.net)

    def update(self, next_image):
        """
        Update bounding box position and size on next_image. Returns True
        beacuse tracking is terminated based on number of frames predicted
        in OpenLabeling, not based on feedback from tracking algorithm (unlike
        the opencv tracking algorithms).
        """
        self.state = SiamRPN_track(self.state, next_image)
        target_pos = self.state["target_pos"]
        target_sz  = self.state["target_sz"]
        bbox = self.pos_to_bbox(target_pos, target_sz)

        return True, bbox

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
