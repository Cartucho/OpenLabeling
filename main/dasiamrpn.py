import torch
import numpy as np
from os.path import realpath, dirname, join
from DaSiamRPN.code.net import SiamRPNvot
from DaSiamRPN.code.run_SiamRPN import SiamRPN_init, SiamRPN_track
from DaSiamRPN.code.utils import get_axis_aligned_bbox, cxy_wh_2_rect

class dasiamrpn(object):
    """
    Wrapper class for incorporating DaSiamRPN into OpenLabeling
    (https://github.com/foolwood/DaSiamRPN,
    https://github.com/Cartucho/OpenLabeling)
    """
    def __init__(self):
        self.net = SiamRPNvot()
        self.net.load_state_dict(torch.load(join(realpath(dirname(__file__)),
            'SiamRPNVOT.model')))
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
        xmin, ymin, w, h = initial_bbox
        cx = int(xmin + w/2)
        cy = int(ymin + h/2)
        target_pos = np.array([cx, cy])
        target_sz  = np.array([w, h])

        return target_pos, target_sz

    def pos_to_bbox(self, target_pos, target_sz):
        w = target_sz[0]
        h = target_sz[1]
        xmin = int(target_pos[0] - w/2)
        ymin = int(target_pos[1] - h/2)

        return xmin, ymin, w, h
