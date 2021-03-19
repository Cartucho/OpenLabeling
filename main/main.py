#!/bin/python
import argparse
import glob
import json, orjson, ujson
import os
import re
import time
import cv2
import numpy as np
from tqdm import tqdm
from shutil import copyfile
import torch
from centernet_better.train import CenterNetBetterModule
import tkinter as tk
import sys


# load class list
def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line

with open('class_list.txt') as f:
    CLASS_LIST = list(nonblank_lines(f))

CLASSES_INDEX = {}
for i in range(len(CLASS_LIST)):
    CLASSES_INDEX[CLASS_LIST[i]] = i

#print(CLASS_LIST)
last_class_index = len(CLASS_LIST) - 1

DELAY = 20 # keyboard delay (in milliseconds)
WITH_QT = False
try:
    cv2.namedWindow('Test')
    cv2.displayOverlay('Test', 'Test QT', 500)
    WITH_QT = True
except cv2.error:
    print('-> Please ignore this error message\n')
cv2.destroyAllWindows()

VIDEO_PATH = ''
CAPTURE = None
to_track_objects = []

parser = argparse.ArgumentParser(description='Open-source image labeling tool')
parser.add_argument('-i', '--input_dir', default='input', type=str, help='Path to input directory')
parser.add_argument('-o', '--output_dir', default='output', type=str, help='Path to output directory')
parser.add_argument('-t', '--thickness', default='2', type=int, help='Bounding box and cross line thickness')

parser.add_argument('--tracker', default='SiamMask', type=str, help="tracker_type being used: ['SiamMask']")
parser.add_argument('-n', '--n_frames', default='10000000', type=int, help='number of frames to track object for')
parser.add_argument('--detector', default='../object_detection/crow/epoch=46-step=17342.ckpt', type=str, help='Detector checkpoint dir')
args = parser.parse_args()

model = args.detector
if torch.cuda.is_available():
    detector = CenterNetBetterModule.load_from_checkpoint(model, pretrained_checkpoints_path=None)
    detector = detector.cuda()
else:
    detector = torch.load(model,map_location='cpu').module

class_index = 0
img_index = 0
img = None
img_objects = []
show_curr_tracked = False
current_data = None
non_track_objects = []
redo_tracking_objects = []

INPUT_DIR  = args.input_dir
OUTPUT_DIR = args.output_dir
N_FRAMES   = args.n_frames
TRACKER_TYPE = args.tracker

if TRACKER_TYPE == "DASIAMRPN":
    from dasiamrpn import dasiamrpn

if TRACKER_TYPE == "SiamMask":
    from siammask import SiamMask


WINDOW_NAME    = 'OpenLabeling'
TRACKBAR_IMG   = 'Image'
TRACKBAR_CLASS = 'Class'

annotation_formats = {'Annotations' : '.txt'}
TRACKER_DIR = os.path.join(OUTPUT_DIR, '.tracker')

labeling_file = {} # dictionary which takes for every img entry: anchor_id,xmin,ymin,w,h,class_id,class_name
labeling_file_dir = None
curr_anchor_id = 0

# selected bounding box
prev_was_double_click = False
prev_was_triple_click = False
is_bbox_selected = False
selected_bbox = -1
LINE_THICKNESS = args.thickness

mouse_x = 0
mouse_y = 0
point_1 = (-1, -1)
point_2 = (-1, -1)

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''


# Check if a point belongs to a rectangle
def pointInRect(pX, pY, rX_left, rY_top, rX_right, rY_bottom):
    return rX_left <= pX <= rX_right and rY_top <= pY <= rY_bottom



# Class to deal with bbox resizing
class dragBBox:
    '''
        LT -- MT -- RT
        |            |
        LM          RM
        |            |
        LB -- MB -- RB
    '''

    # Size of resizing anchors (depends on LINE_THICKNESS)
    sRA = LINE_THICKNESS * 2

    # Object being dragged
    selected_object = None

    # Flag indicating which resizing-anchor is dragged
    anchor_being_dragged = None

    '''
    \brief This method is used to check if a current mouse position is inside one of the resizing anchors of a bbox
    '''
    @staticmethod
    def check_point_inside_resizing_anchors(eX, eY, obj):
        anchor_id, x_left, y_top, x_right, y_bottom,class_id,class_name = obj
        # first check if inside the bbox region (to avoid making 8 comparisons per object)
        if pointInRect(eX, eY,
                        x_left - dragBBox.sRA,
                        y_top - dragBBox.sRA,
                        x_right + dragBBox.sRA,
                        y_bottom + dragBBox.sRA):

            anchor_dict = get_anchors_rectangles(x_left, y_top, x_right, y_bottom)
            for anchor_key in anchor_dict:
                rX_left, rY_top, rX_right, rY_bottom = anchor_dict[anchor_key]
                if pointInRect(eX, eY, rX_left, rY_top, rX_right, rY_bottom):
                    dragBBox.anchor_being_dragged = anchor_key
                    break

    '''
    \brief This method is used to select an object if one presses a resizing anchor
    '''
    @staticmethod
    def handler_left_mouse_down(eX, eY, obj):
        dragBBox.check_point_inside_resizing_anchors(eX, eY, obj)
        if dragBBox.anchor_being_dragged is not None:
            dragBBox.selected_object = obj

    @staticmethod
    def handler_mouse_move(eX, eY):
        if dragBBox.selected_object is not None:
            anchor_id, x_left, y_top, x_right, y_bottom,class_id,class_name= dragBBox.selected_object

            # Do not allow the bbox to flip upside down (given a margin)
            margin = 3 * dragBBox.sRA
            change_was_made = False

            if dragBBox.anchor_being_dragged[0] == "L":
                # left anchors (LT, LM, LB)
                if eX < x_right - margin:
                    x_left = eX
                    change_was_made = True
            elif dragBBox.anchor_being_dragged[0] == "R":
                # right anchors (RT, RM, RB)
                if eX > x_left + margin:
                    x_right = eX
                    change_was_made = True

            if dragBBox.anchor_being_dragged[1] == "T":
                # top anchors (LT, RT, MT)
                if eY < y_bottom - margin:
                    y_top = eY
                    change_was_made = True
            elif dragBBox.anchor_being_dragged[1] == "B":
                # bottom anchors (LB, RB, MB)
                if eY > y_top + margin:
                    y_bottom = eY
                    change_was_made = True

            if change_was_made:
                action = "resize_bbox:{}:{}:{}:{}".format(x_left, y_top, x_right, y_bottom)
                edit_bbox(dragBBox.selected_object, action)
                # update the selected bbox
                dragBBox.selected_object = [anchor_id,x_left, y_top, x_right, y_bottom,class_id,class_name]

    '''
    \brief This method will reset this class
     '''
    @staticmethod
    def handler_left_mouse_up(eX, eY):
        if dragBBox.selected_object is not None:
            dragBBox.selected_object = None
            dragBBox.anchor_being_dragged = None

def display_text(text, time):
    if WITH_QT:
        cv2.displayOverlay(WINDOW_NAME, text, time)
    else:
        print(text)

def set_img_index(x):
    global img_index, img
    img_index = x
    img_path = IMAGE_PATH_LIST[img_index]
    img = cv2.imread(img_path)
    
    # CAPTURE.set(cv2.CAP_PROP_POS_FRAMES,img_index)
    # _,img = CAPTURE.read()

    text = 'Showing image {}/{}, path: {}'.format(str(img_index), str(last_img_index), img_path)
    display_text(text, 1000)


def set_class_index(x):
    global class_index
    class_index = x
    text = 'Selected class {}/{} -> {}'.format(str(class_index), str(last_class_index), CLASS_LIST[class_index])
    display_text(text, 3000)


def draw_edges(tmp_img):
    blur = cv2.bilateralFilter(tmp_img, 3, 75, 75)
    edges = cv2.Canny(blur, 150, 250, 3)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Overlap image and edges together
    tmp_img = np.bitwise_or(tmp_img, edges)
    #tmp_img = cv2.addWeighted(tmp_img, 1 - edges_val, edges, edges_val, 0)
    return tmp_img


def decrease_index(current_index, last_index):
    current_index -= 1
    if current_index < 0:
        current_index = last_index
    return current_index


def increase_index(current_index, last_index):
    current_index += 1
    if current_index > last_index:
        current_index = 0
    return current_index


def draw_line(img, x, y, height, width, color):
    cv2.line(img, (x, 0), (x, height), color, LINE_THICKNESS)
    cv2.line(img, (0, y), (width, y), color, LINE_THICKNESS)



def findIndex(obj_to_find):
    #return [(ind, img_objects[ind].index(obj_to_find)) for ind in xrange(len(img_objects)) if item in img_objects[ind]]
    ind = -1

    ind_ = 0
    for listElem in img_objects:
        if listElem == obj_to_find:
            ind = ind_
            return ind
        ind_ = ind_+1

    return ind


def get_anchors_rectangles(xmin, ymin, xmax, ymax):
    anchor_list = {}

    mid_x = (xmin + xmax) / 2
    mid_y = (ymin + ymax) / 2

    L_ = [xmin - dragBBox.sRA, xmin + dragBBox.sRA]
    M_ = [mid_x - dragBBox.sRA, mid_x + dragBBox.sRA]
    R_ = [xmax - dragBBox.sRA, xmax + dragBBox.sRA]
    _T = [ymin - dragBBox.sRA, ymin + dragBBox.sRA]
    _M = [mid_y - dragBBox.sRA, mid_y + dragBBox.sRA]
    _B = [ymax - dragBBox.sRA, ymax + dragBBox.sRA]

    anchor_list['LT'] = [L_[0], _T[0], L_[1], _T[1]]
    anchor_list['MT'] = [M_[0], _T[0], M_[1], _T[1]]
    anchor_list['RT'] = [R_[0], _T[0], R_[1], _T[1]]
    anchor_list['LM'] = [L_[0], _M[0], L_[1], _M[1]]
    anchor_list['RM'] = [R_[0], _M[0], R_[1], _M[1]]
    anchor_list['LB'] = [L_[0], _B[0], L_[1], _B[1]]
    anchor_list['MB'] = [M_[0], _B[0], M_[1], _B[1]]
    anchor_list['RB'] = [R_[0], _B[0], R_[1], _B[1]]

    return anchor_list


def draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, color):
    anchor_dict = get_anchors_rectangles(xmin, ymin, xmax, ymax)
    for anchor_key in anchor_dict:
        x1, y1, x2, y2 = anchor_dict[anchor_key]
        cv2.rectangle(tmp_img, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
    return tmp_img

def draw_bboxes_from_dict(tmp_img, img_path, width=0, height=0):
    global img_objects, is_bbox_selected, selected_bbox
    img_objects = []
    ann_path = None
    
    # Drawing bounding boxes from the YOLO files
    # ann_path = next(path for path in annotation_paths if 'YOLO_darknet' in path)
    if labeling_file.get(img_path, None)!=None:
        objs = labeling_file[img_path]
        nested_list = any(isinstance(i, list) for i in objs)
        if nested_list:
            for idx,obj in enumerate(objs):
                if len(obj)==0:
                    continue
                anchor_id,xmin, ymin, w, h, class_index,class_name = obj
                # class_index = CLASSES_INDEX[class_name]
                # class_name = CLASS_LIST[class_index]
                xmax = xmin+w
                ymax = ymin+h

                img_objects.append([anchor_id,xmin, ymin, xmax, ymax,class_index,CLASS_LIST[class_index]])
                color = class_rgb[class_index].tolist()
                # draw bbox
                cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), color, LINE_THICKNESS)
                # draw resizing anchors if the object is selected
                if is_bbox_selected:
                    if idx == selected_bbox:
                        tmp_img = draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, color)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(tmp_img, class_name+str(anchor_id), (xmin, ymin - 5), font, 0.6, color, LINE_THICKNESS, cv2.LINE_AA)
        else:
            if objs == []:
                return tmp_img
            anchor_id, xmin, ymin, w, h,class_index, class_name = objs
            # class_name = CLASS_LIST[class_index]
            xmax = xmin+w
            ymax = ymin+h


            img_objects.append([anchor_id,xmin, ymin, xmax, ymax,class_index,CLASS_LIST[class_index]])
            color = class_rgb[class_index].tolist()
            # draw bbox
            cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), color, LINE_THICKNESS)
            # draw resizing anchors if the object is selected
            if is_bbox_selected:
                if 0 == selected_bbox:
                    tmp_img = draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, color)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(tmp_img, class_name+str(anchor_id), (xmin, ymin - 5), font, 0.6, color, LINE_THICKNESS, cv2.LINE_AA)
        
            
    return tmp_img


def get_bbox_area(x1, y1, x2, y2):
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width*height


def set_selected_bbox(set_class):
    global is_bbox_selected, selected_bbox
    smallest_area = -1
    # if clicked inside multiple bboxes selects the smallest one
    for idx, obj in enumerate(img_objects):
        anchor_id, x1, y1, x2, y2,class_id,class_name = obj
        x1 = x1 - dragBBox.sRA
        y1 = y1 - dragBBox.sRA
        x2 = x2 + dragBBox.sRA
        y2 = y2 + dragBBox.sRA
        if pointInRect(mouse_x, mouse_y, x1, y1, x2, y2):
            is_bbox_selected = True
            tmp_area = get_bbox_area(x1, y1, x2, y2)
            if tmp_area < smallest_area or smallest_area == -1:
                smallest_area = tmp_area
                selected_bbox = idx
                if set_class:
                    # set class to the one of the selected bounding box
                    cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_id)


def is_mouse_inside_delete_button():
    for idx, obj in enumerate(img_objects):
        if idx == selected_bbox:
            anchor_id, x1, y1, x2, y2,class_id,class_name = obj
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            if pointInRect(mouse_x, mouse_y, x1_c, y1_c, x2_c, y2_c):
                return True
    return False

def is_mouse_inside_tracked_button():
    for idx, obj in enumerate(img_objects):
        if idx == selected_bbox:
            anchor_id, x1, y1, x2, y2,class_id,class_name = obj
            x1_c, y1_c, x2_c, y2_c = get_tracked_icon(x1, y1, x2, y2)
            if pointInRect(mouse_x, mouse_y, x1_c, y1_c, x2_c, y2_c):
                return True
    return False

def inside_bbox(x, y, bbox):
    x1, y1, w, h = bbox
    x2, y2 = x1+w, y1+h
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False


def edit_bbox(obj_to_edit, action):
    ''' action = `delete`
                 `change_class:new_class_index`
                 `resize_bbox:new_x_left:new_y_top:new_x_right:new_y_bottom`
    '''
    global selected_bbox
    global curr_anchor_id
    if 'change_class' in action:
        new_class_index = int(action.split(':')[1])
    elif 'change_trackid' in action:
        new_trackid = int(action.split(':')[1])
    elif 'resize_bbox' in action:
        new_x_left = max(0, int(action.split(':')[1]))
        new_y_top = max(0, int(action.split(':')[2]))
        new_x_right = min(width, int(action.split(':')[3]))
        new_y_bottom = min(height, int(action.split(':')[4]))

    # 1. initialize bboxes_to_edit_dict
    #    (we use a dict since a single label can be associated with multiple ones in videos)
    bboxes_to_edit_dict = {}
    current_img_path = IMAGE_PATH_LIST[img_index]
    bboxes_to_edit_dict[current_img_path] = obj_to_edit

    # 2. add elements to bboxes_to_edit_dict
    '''
        If the bbox is in the json file then it was used by the video Tracker, hence,
        we must also edit the next predicted bboxes associated to the same `anchor_id`.
    '''
    # if `current_img_path` is a frame from a video
    is_from_video, video_name = is_frame_from_video(current_img_path)
    if is_from_video:
        # get json file corresponding to that video
        json_file_path = '{}.json'.format(os.path.join(TRACKER_DIR, video_name))
        file_exists, json_file_data = get_json_file_data(json_file_path)
        # if json file exists
        if file_exists:
            # match obj_to_edit with the corresponding json object
            frame_data_dict = json_file_data['frame_data_dict']
            json_object_list = get_json_file_object_list(current_img_path, frame_data_dict)
            obj_matched = get_json_object_dict(obj_to_edit, json_object_list)
            # if match found
            if obj_matched is not None:
                # get this object's anchor_id
                anchor_id = obj_matched['anchor_id']

                frame_path_list = get_next_frame_path_list(video_name, current_img_path)
                frame_path_list.insert(0, current_img_path)

                if 'change_class' in action:
                    # add also the previous frames
                    prev_path_list = get_prev_frame_path_list(video_name, current_img_path)
                    frame_path_list = prev_path_list + frame_path_list

                # update json file if contain the same anchor_id
                for frame_path in frame_path_list:
                    json_object_list = get_json_file_object_list(frame_path, frame_data_dict)
                    if (('delete_recursive' in action or 'change_class' in action or 'change_trackid' in action) 
                            and int(img_path.split('_')[-1].replace('.jpg',''))>=int(current_img_path.split('_')[-1].replace('.jpg',''))):
                        json_obj = get_json_file_object_by_id(json_object_list, anchor_id) 
                    else:
                        json_obj = get_json_file_object_by_exact_dicription(json_object_list, obj_matched)
                    if json_obj is not None:
                        # edit json file
                        if 'delete' in action:
                            json_object_list.remove(json_obj)
                        elif 'change_class' in action:
                            json_obj['class_index'] = new_class_index
                        elif 'change_trackid' in action:
                            json_obj['anchor_id'] = new_trackid
                        elif 'resize_bbox' in action:
                            json_obj['bbox']['xmin'] = new_x_left
                            json_obj['bbox']['ymin'] = new_y_top
                            json_obj['bbox']['xmax'] = new_x_right
                            json_obj['bbox']['ymax'] = new_y_bottom
                    else:
                        break
                # super slow 
                # save the edited data
                # with open(json_file_path, 'w') as outfile:
                #     json.dump(json_file_data, outfile, sort_keys=True, indent=4)
                #     json.dump(json_file_data, outfile, sort_keys=True, indent=4)

    # 3. loop through bboxes_to_edit_dict and edit the corresponding annotation files
    for path in bboxes_to_edit_dict:
        obj_to_edit = bboxes_to_edit_dict[path]
        class_name = obj_to_edit.pop(-1)
        anchor_id,xmin, ymin, xmax, ymax,class_id = map(int, obj_to_edit)
        w = abs(xmax-xmin)
        h = abs(ymax-ymin)
        obj_to_edit.append(class_name)
        nested_list = any(isinstance(d, list) for d in labeling_file[current_img_path])
        
        if 'delete_recursive' in action:
            labeling_file[current_img_path].pop(selected_bbox)
            if labeling_file[current_img_path]==[]:
                del labeling_file[current_img_path]
            next_frames = get_next_frame_path_list(video_name, current_img_path)
            for frame in next_frames:
                delete_index=-1
                for i,obj in enumerate(labeling_file[frame]):
                # continue_delete = False
                    if obj[0] == anchor_id:
                        delete_index = i
                        break
                if delete_index !=-1:
                    if labeling_file[frame][delete_index][0] == anchor_id:
                        labeling_file[frame].pop(delete_index)
                        if labeling_file[frame]==[]:
                            del labeling_file[frame]
                else:
                    break
                
        elif 'delete' in action:
            labeling_file[current_img_path].pop(selected_bbox)
            if labeling_file[current_img_path]==[]:
                del labeling_file[current_img_path]


        elif 'change_class' in action:
            if nested_list:
                labeling_file[current_img_path][selected_bbox] = [curr_anchor_id,xmin,ymin,w,h,new_class_index,CLASS_LIST[new_class_index]]
            else:
                labeling_file[current_img_path]= [curr_anchor_id,xmin,ymin,w,h,new_class_index,CLASS_LIST[new_class_index]]
            next_frames = get_next_frame_path_list(video_name, current_img_path)
            for frame in next_frames:
                if nested_list:
                    if labeling_file.get(frame,None)!=None:
                        for i in range(len(labeling_file[frame])):
                            if labeling_file[frame][i][0] == anchor_id:
                                next_anchor_id,next_xmin,next_ymin,next_w,next_h,next_classid, next_class_name = labeling_file[frame][i]
                                labeling_file[frame][i] = [curr_anchor_id,next_xmin,next_ymin,next_w,next_h,new_class_index,CLASS_LIST[new_class_index]]
                else:
                    if labeling_file[frame][0]==anchor_id:
                        next_anchor_id,next_xmin,next_ymin,next_w,next_h,next_classid, next_class_name = labeling_file[frame]
                        labeling_file[frame]= [curr_anchor_id,next_xmin,next_ymin,next_w,next_h,new_class_index,CLASS_LIST[new_class_index]]
            curr_anchor_id+=1
        elif 'change_trackid' in action:
            if nested_list:
                labeling_file[current_img_path][selected_bbox] = [new_trackid,xmin,ymin,w,h,CLASSES_INDEX[class_name],class_name]
            else:
                labeling_file[current_img_path]= [new_trackid,xmin,ymin,w,h,CLASSES_INDEX[class_name],class_name]
            next_frames = get_next_frame_path_list(video_name, current_img_path)
            for frame in next_frames:
                if nested_list:
                    if labeling_file.get(frame,None)!=None:
                        for i in range(len(labeling_file[frame])):
                            if labeling_file[frame][i][0] == anchor_id:
                                next_anchor_id,next_xmin,next_ymin,next_w,next_h,next_classid, next_class_name = labeling_file[frame][i]
                                labeling_file[frame][i] = [new_trackid,next_xmin,next_ymin,next_w,next_h,next_classid,next_class_name]
                else:
                    if labeling_file[frame][0]==anchor_id:
                        next_anchor_id,next_xmin,next_ymin,next_w,next_h,next_classid, next_class_name = labeling_file[frame]
                        labeling_file[frame]= [new_trackid,next_xmin,next_ymin,next_w,next_h,next_classid,next_class_name]
            curr_anchor_id+=1
        elif 'resize_bbox' in action:
            new_w = abs(new_x_left-new_x_right)
            new_h = abs(new_y_top-new_y_bottom)
            if nested_list:
                labeling_file[current_img_path][selected_bbox]= [anchor_id,new_x_left,new_y_top,new_w,new_h,class_index,class_name]
            else:
                labeling_file[current_img_path] = [anchor_id,new_x_left,new_y_top,new_w,new_h,class_index,class_name]
    # save_darklabel_txt(labeling_file_dir)

            



def mouse_listener(event, x, y, flags, param):
    try:
        # mouse callback function
        global is_bbox_selected, prev_was_double_click, prev_was_triple_click, mouse_x, mouse_y, point_1, point_2

        set_class = True
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_x = x
            mouse_y = y
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            prev_was_double_click = True
            #print('Double click')
            point_1 = (-1, -1)
            # if clicked inside a bounding box we set that bbox
            set_selected_bbox(set_class)
        # By AlexeyGy: delete via right-click
        elif event == cv2.EVENT_RBUTTONDOWN:
            set_class = False
            set_selected_bbox(set_class)
            if is_bbox_selected:
                obj_to_edit = img_objects[selected_bbox]
                
                # if keyboard.is_pressed('r'):
                #     print('remove with r')
                edit_bbox(obj_to_edit, 'delete')
                is_bbox_selected = False
        elif event == cv2.EVENT_MBUTTONDOWN:
            if is_bbox_selected:
                c_id, x1,y1,x2,y2,_,_= img_objects[selected_bbox]
                if pointInRect(x,y,x1,y1,x2,y2):
                # if inside_bbox(x, y, img_objects[selected_bbox]):
                # dragBBox.handler_left_mouse_down(x, y, img_objects[selected_bbox])
                    root = tk.Tk()
                    root.withdraw()
                    USER_INP = tk.simpledialog.askstring(title="TrackId",
                                    prompt=f"Current TrackId {c_id}, New TrackId:")

                    obj_to_edit = img_objects[selected_bbox]
                    edit_bbox(obj_to_edit, 'change_trackid:{}'.format(USER_INP))
                    # print("TrackId", USER_INP)
                    root.destroy()


        elif event == cv2.EVENT_LBUTTONDOWN:
           
            if prev_was_double_click:
                #print('Finish double click')
                prev_was_double_click = False
                prev_was_triple_click = True
            else:
                #print('Normal left click')

                # Check if mouse inside on of resizing anchors of the selected bbox
                if is_bbox_selected:
                    dragBBox.handler_left_mouse_down(x, y, img_objects[selected_bbox])

                if dragBBox.anchor_being_dragged is None:
                    if point_1[0] == -1:
                        if is_bbox_selected:
                            if is_mouse_inside_delete_button():
                                # set_selected_bbox(set_class)
                                obj_to_edit = img_objects[selected_bbox]
                                edit_bbox(obj_to_edit, 'delete_recursive')
                            if is_mouse_inside_tracked_button():
                                obj_to_edit = img_objects[selected_bbox]
                                if obj_to_edit  in object_list:
                                    if obj_to_edit in redo_tracking_objects:
                                        redo_tracking_objects.remove(obj_to_edit)
                                    non_track_objects.append(obj_to_edit)
                                else:
                                    if obj_to_edit in non_track_objects:
                                        non_track_objects.remove(obj_to_edit)
                                    redo_tracking_objects.append(obj_to_edit)
                            is_bbox_selected = False
                        else:
                            # first click (start drawing a bounding box or delete an item)

                            point_1 = (x, y)
                    else:
                        # minimal size for bounding box to avoid errors
                        threshold = 5
                        if abs(x - point_1[0]) > threshold or abs(y - point_1[1]) > threshold:
                            # second click
                            point_2 = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if dragBBox.anchor_being_dragged is not None:
                dragBBox.handler_left_mouse_up(x, y)
    except Exception:
        pass




def get_close_icon(x1, y1, x2, y2):
    percentage = 0.05
    height = -1
    while height < 15 and percentage < 1.0:
        height = int((y2 - y1) * percentage)
        percentage += 0.1
    return (x2 - height), y1, x2, (y1 + height)

def get_tracked_icon(x1,y1,x2,y2):
    percentage = 0.05
    height = -1
    while height < 10 and percentage < 0.50:
        height = int((y2 - y1) * percentage)
        percentage += 0.05
    return x1, y1, (x1+height), (y1 + height)



def draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c):
    red = (0,0,255)
    cv2.rectangle(tmp_img, (x1_c + 1, y1_c - 1), (x2_c, y2_c), red, -1)
    white = (255, 255, 255)
    cv2.line(tmp_img, (x1_c, y1_c), (x2_c, y2_c), white, 2)
    cv2.line(tmp_img, (x1_c, y2_c), (x2_c, y1_c), white, 2)
    return tmp_img

def draw_tracked_icon(tmp_img, x1_c, y1_c, x2_c, y2_c):
    green = (0,255,0)
    cv2.rectangle(tmp_img, (x1_c + 1, y1_c - 1), (x2_c, y2_c), green, -1)
    white = (255, 255, 255)
    # cv2.line(tmp_img, (x1_c, y1_c), (x2_c, y2_c), white, 2)
    # cv2.line(tmp_img, (x1_c, y2_c), (x2_c, y1_c), white, 2)
    return tmp_img

def draw_tracked_icon_grey(tmp_img, x1_c, y1_c, x2_c, y2_c):
    grey = (192,192,192)
    cv2.rectangle(tmp_img, (x1_c + 1, y1_c - 1), (x2_c, y2_c), grey, -1)
    white = (255, 255, 255)
    # cv2.line(tmp_img, (x1_c, y1_c), (x2_c, y2_c), white, 2)
    # cv2.line(tmp_img, (x1_c, y2_c), (x2_c, y1_c), white, 2)
    return tmp_img

def draw_info_bb_selected(tmp_img):
    for idx, obj in enumerate(img_objects):
        anchor_id, x1, y1, x2, y2,class_index,class_name = obj
        if idx == selected_bbox:
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c)
    return tmp_img


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def convert_video_to_images(video_path, n_frames, desired_img_format):
    # create folder to store images (if video was not converted to images already)
    file_path, file_extension = os.path.splitext(video_path)
    # append extension to avoid collision of videos with same name
    # e.g.: `video.mp4`, `video.avi` -> `video_mp4/`, `video_avi/`
    file_extension = file_extension.replace('.', '_')
    file_path += file_extension
    video_name_ext = os.path.basename(file_path)
    if not os.path.exists(file_path):
        print(' Converting video to individual frames...')
        cap = cv2.VideoCapture(video_path)
        os.makedirs(file_path)
        # read the video
        for i in tqdm(range(n_frames)):
            if not cap.isOpened():
                break
            # capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # save each frame (we use this format to avoid repetitions)
                frame_name =  '{}_{}{}'.format(video_name_ext, i, desired_img_format)
                frame_path = os.path.join(file_path, frame_name)
                cv2.imwrite(frame_path, frame)
        # release the video capture object
        cap.release()
    return file_path, video_name_ext


def overlap_percent_bbox(obj1,obj2):
    # obj1 = obj[1:5]
    # obj2 = obj_[1:5]
    SA=(obj1[0]-obj1[2])*(obj1[1]-obj1[3])
    SB=(obj2[0]-obj2[2])*(obj2[1]-obj2[3])
    SI=(max(0,-(max(obj1[0],obj2[0])-min(obj1[2],obj2[2])))) * (max(0,-(max(obj1[1],obj2[1])-min(obj1[3],obj2[3]))))
    # A_overlap=(max(obj1[0],obj2[0])-min(obj1[2],obj2[2]))*(max(obj1[1],obj2[1])-min(obj1[3],obj2[3]))
    # p_overlap = abs(A_overlap/(A1+A2-A_overlap))
    # p_overlap = abs(A_overlap/A1)
    SU = SA+SB-SI

    return SI/SU

def get_annotation_paths(img_path, annotation_formats):
    annotation_paths = []
    for ann_dir, ann_ext in annotation_formats.items():
        new_path = os.path.join(OUTPUT_DIR, ann_dir)
        new_path = os.path.join(new_path, os.path.basename(os.path.normpath(img_path))) #img_path.replace(INPUT_DIR, new_path, 1)
        pre_path, img_ext = os.path.splitext(new_path)
        new_path = new_path.replace(img_ext, ann_ext, 1)
        annotation_paths.append(new_path)
    return annotation_paths



def read_darklabel_file(file_dir,video_name):
    # frame_number = img_path.split('_')[-1].replace('.jpg','')
    f = open(file_dir,'r')
    relevant_objs = []
    
    for line in f:
        objs = line.split('\n')[0].split(',')
        # if int(objs[0]) == int(frame_number):
        frame_number = objs.pop(0)
        class_name = objs.pop(-1)
        objs = list(map(int, objs))
        objs.append(CLASSES_INDEX[class_name])
        objs.append(class_name)
        relevant_objs.append(objs)
        key = video_name+frame_number+'.jpg'
        if labeling_file.get(key,None) == None:
            labeling_file[key] = [objs]
        else:
            labeling_file[key].append(objs)
        # labeling_file[]
    # curr_anchor_id = max(relevant_objs,key=lambda o: o[0])
    # return relevant_objs

def update_bounding_box(frame_path,anchor_id,class_index,xmin,ymin,xmax,ymax):
    w= abs(xmax-xmin)
    h = abs(ymax-ymin)
    if labeling_file.get(frame_path,None)==None:
        labeling_file[frame_path] = [[anchor_id, xmin,ymin,w,h,class_index,CLASS_LIST[class_index]]]
    else:
        if labeling_file[frame_path] == []:
            labeling_file[frame_path] = [[anchor_id, xmin,ymin,w,h,class_index,CLASS_LIST[class_index]]]
        else:
           labeling_file[frame_path].append([anchor_id, xmin,ymin,w,h,class_index,CLASS_LIST[class_index]])    



def save_darklabel_txt(labeling_file_dir):
    open(labeling_file_dir, "w")
    with open(labeling_file_dir, "a+") as f:
        for k in labeling_file.keys():
            objs = labeling_file[k]
            frame_number = k.split('_')[-1].replace('.jpg','')
            nested_list = any(isinstance(i, list) for i in objs)
            if nested_list:
                for obj in objs:
                    index = obj.pop(-2)
                    output_line = frame_number + ',' + ','.join(str(e) for e in obj)
                    obj.insert(-1,index)
                    f.write(output_line)
                    f.write("\n")
            else:
                if objs == []:
                    continue
                index = objs.pop(-2)
                output_line = frame_number + ',' + ','.join(str(e) for e in objs)
                objs.insert(-1,index)
                f.write(output_line)
                f.write("\n")


def set_max_anchor():
    global curr_anchor_id
    max_anchor = 0
    for k in labeling_file.keys():
        objs = labeling_file[k]
        nested_list = any(isinstance(i, list) for i in objs)
        if nested_list:
            for idx,obj in enumerate(objs):
                if obj[0]>max_anchor:
                    max_anchor=obj[0]
        else:
            if objs == []:
                continue
            if objs[0]>max_anchor:
                max_anchor=objs[0]

    curr_anchor_id=max_anchor+1
        




def is_frame_from_video(img_path):
    for video_name in VIDEO_NAME_DICT:
        video_dir = os.path.join(INPUT_DIR, video_name)
        if os.path.dirname(img_path) == video_dir:
            # image belongs to a video
            return True, video_name
    return False, None


def get_json_file_data(json_file_path):
    if os.path.isfile(json_file_path):
        with open(json_file_path) as f:
            # data = json.load(f)
            data = json.load(f)
            return True, data
    else:
        return False, {'n_anchor_ids':0, 'frame_data_dict':{}}

def update_json_anchor_id():
    is_from_video, video_name = is_frame_from_video(img_path)
    if is_from_video:
        # get list of objects associated to that frame
        object_list = img_objects[:]
        # remove the objects in that frame that are already in the `.json` file
        json_file_path = '{}.json'.format(os.path.join(TRACKER_DIR, video_name))
        if os.path.isfile(json_file_path):
            with open(json_file_path) as f:
                data = json.load(f)
                anchor_id = data['n_anchor_ids']
                data.update({'n_anchor_ids': (anchor_id + 1)})


def get_prev_frame_path_list(video_name, img_path):
    first_index = VIDEO_NAME_DICT[video_name]['first_index']
    last_index = VIDEO_NAME_DICT[video_name]['last_index']
    img_index = IMAGE_PATH_LIST.index(img_path)
    return IMAGE_PATH_LIST[first_index:img_index]


def get_next_frame_path_list(video_name, img_path):
    first_index = VIDEO_NAME_DICT[video_name]['first_index']
    last_index = VIDEO_NAME_DICT[video_name]['last_index']
    img_index = IMAGE_PATH_LIST.index(img_path)
    return IMAGE_PATH_LIST[(img_index + 1):last_index]


def get_json_object_dict(obj, json_object_list):
    if len(json_object_list) > 0:
        class_name = obj.pop(-1)
        anchor_idx,xmin, ymin, xmax, ymax,class_index = map(int, obj)
        obj.append(class_name)
        for d in json_object_list:
                    if ( d['class_index'] == class_index and
                         d['bbox']['xmin'] == xmin and
                         d['bbox']['ymin'] == ymin and
                         d['bbox']['xmax'] == xmax and
                         d['bbox']['ymax'] == ymax ) :
                        return d
    return None


def remove_already_tracked_objects(object_list, img_path, json_file_data):
    frame_data_dict = json_file_data['frame_data_dict']
    json_object_list = get_json_file_object_list(img_path, frame_data_dict)
    # copy the list since we will be deleting elements without restarting the loop
    temp_object_list = object_list[:]
    for obj in temp_object_list:
        obj_dict = get_json_object_dict(obj, json_object_list)
        if obj_dict is not None and obj not in redo_tracking_objects:
            object_list.remove(obj)
            # json_object_list.remove(obj_dict)
        elif obj in non_track_objects:
            object_list.remove(obj)
 
    return object_list


def get_json_file_object_by_id(json_object_list, anchor_id):
    for obj_dict in json_object_list:
        if obj_dict['anchor_id'] == anchor_id:
            return obj_dict
    return None

def get_json_file_object_by_exact_dicription(json_object_list, object_to_edit):
    for obj_dict in json_object_list:
        if obj_dict ==object_to_edit:
            return obj_dict
        
    return None


def get_json_file_object_list(img_path, frame_data_dict):
    object_list = []
    if img_path in frame_data_dict:
        object_list = frame_data_dict[img_path]
    return object_list


def json_file_add_object(frame_data_dict, img_path, anchor_id, pred_counter, obj):
    object_list = get_json_file_object_list(img_path, frame_data_dict)
    anchor_id_, xmin, ymin, xmax, ymax,class_index,class_name = obj

    bbox = {
      'xmin': xmin,
      'ymin': ymin,
      'xmax': xmax,
      'ymax': ymax
    }

    temp_obj = {
      'anchor_id': anchor_id,
      'prediction_index': pred_counter,
      'class_index': class_index,
      'bbox': bbox
    }

    object_list.append(temp_obj)
    frame_data_dict[img_path] = object_list

    return frame_data_dict


class LabelTracker():
    ''' Special thanks to Rafael Caballero Gonzalez '''
    # extract the OpenCV version info, e.g.:
    # OpenCV 3.3.4 -> [major_ver].[minor_ver].[subminor_ver]
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # TODO: press ESC to stop the tracking process

    def __init__(self, tracker_type, init_frame, next_frame_path_list):
        tracker_types = ['CSRT', 'KCF','MOSSE', 'MIL', 'BOOSTING', 'MEDIANFLOW', 'TLD', 'GOTURN', 'DASIAMRPN','SiamMask']
        ''' Recomended tracker_type:
              KCF -> KCF is usually very good (minimum OpenCV 3.1.0)
              CSRT -> More accurate than KCF but slightly slower (minimum OpenCV 3.4.2)
              MOSSE -> Less accurate than KCF but very fast (minimum OpenCV 3.4.1)
        '''
        self.tracker_type = tracker_type
        # -- TODO: remove this if I assume OpenCV version > 3.4.0
        if tracker_type == tracker_types[0] or tracker_type == tracker_types[2]:
            if int(self.major_ver == 3) and int(self.minor_ver) < 4:
                self.tracker_type = tracker_types[1] # Use KCF instead of CSRT or MOSSE
        # --
        self.init_frame = init_frame
        self.next_frame_path_list = next_frame_path_list

        self.img_h, self.img_w = init_frame.shape[:2]


    def call_tracker_constructor(self, tracker_type):
        if tracker_type == 'DASIAMRPN':
            tracker = dasiamrpn()
        elif tracker_type == 'SiamMask':
            tracker = SiamMask()
        else:
            # -- TODO: remove this if I assume OpenCV version > 3.4.0
            if int(self.major_ver == 3) and int(self.minor_ver) < 3:
                #tracker = cv2.Tracker_create(tracker_type)
                pass
            # --
            else:
                try:
                    tracker = cv2.TrackerKCF_create()
                except AttributeError as error:
                    print(error)
                    print('\nMake sure that OpenCV contribute is installed: opencv-contrib-python\n')
                if tracker_type == 'CSRT':
                    tracker = cv2.TrackerCSRT_create()
                elif tracker_type == 'KCF':
                    tracker = cv2.TrackerKCF_create()
                elif tracker_type == 'MOSSE':
                    tracker = cv2.TrackerMOSSE_create()
                elif tracker_type == 'MIL':
                    tracker = cv2.TrackerMIL_create()
                elif tracker_type == 'BOOSTING':
                    tracker = cv2.TrackerBoosting_create()
                elif tracker_type == 'MEDIANFLOW':
                    tracker = cv2.TrackerMedianFlow_create()
                elif tracker_type == 'TLD':
                    tracker = cv2.TrackerTLD_create()
                elif tracker_type == 'GOTURN':
                    tracker = cv2.TrackerGOTURN_create()
                
        return tracker


    def start_tracker(self, json_file_data, json_file_path, img_path, obj, color, annotation_formats):
        global img_index
        tracker = self.call_tracker_constructor(self.tracker_type)
        # anchor_id = json_file_data['n_anchor_ids']
        anchor_id = obj[0]
        frame_data_dict = json_file_data['frame_data_dict']

        pred_counter = 0
        frame_data_dict = json_file_add_object(frame_data_dict, img_path, anchor_id, pred_counter, obj)
        # tracker bbox format: xmin, xmax, w, h
        xmin, ymin, xmax, ymax = obj[1:5]
        initial_bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
        tracker.init(self.init_frame, initial_bbox)
        # final_frame = None
        for frame_path in self.next_frame_path_list:
            next_image = cv2.imread(frame_path)
            img_index = increase_index(img_index, last_img_index)
            
            # cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)
            
            
            # final_frame = 
            # get the new bbox prediction of the object
            success, bbox = tracker.update(next_image.copy())
            if pred_counter >= N_FRAMES:
                success = False
            if success:
                
                pred_counter += 1
                # xmin, ymin, w, h = map(int, bbox)
                xmin,ymin,w,h=bbox
                xmax = xmin + w
                ymax = ymin + h
                obj = [anchor_id,xmin, ymin, xmax, ymax,int(class_index),class_name]
                frame_data_dict = json_file_add_object(frame_data_dict, frame_path, anchor_id, pred_counter, obj)
                cv2.rectangle(next_image, (xmin, ymin), (xmax, ymax), color, LINE_THICKNESS+1)
                # save prediction
                annotation_paths = get_annotation_paths(frame_path, annotation_formats)
                # save_bounding_box(annotation_paths, class_index, (xmin, ymin), (xmax, ymax), self.img_w, self.img_h)
                update_bounding_box(frame_path,anchor_id,int(class_index),xmin,ymin,xmax,ymax)

                # show prediction
                next_image = draw_bboxes_from_dict(next_image,frame_path)
                cv2.imshow(WINDOW_NAME, next_image)
                pressed_key = cv2.waitKey(DELAY)
                if pressed_key == ord('x'):
                    
                    break
            else:
                break
        set_img_index(img_index)
        cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)

        # json_file_data.update({'n_anchor_ids': (anchor_id + 1)})
        # curr_anchor_id
        # save the updated data
        # copyfile(json_file_path, json_file_path[:-5]+'_backup.json')
        # with open(json_file_path, 'w') as outfile:
        #     json.dump(json_file_data, outfile, sort_keys=True, indent=4)
        # save_darklabel_txt(labeling_file_dir)


def complement_bgr(color):
    lo = min(color)
    hi = max(color)
    k = lo + hi
    return tuple(k - u for u in color)




# change to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    # load all images and videos (with multiple extensions) from a directory using OpenCV
    IMAGE_PATH_LIST = []
    VIDEO_NAME_DICT = {}
    for f in sorted(os.listdir(INPUT_DIR), key = natural_sort_key):
        f_path = os.path.join(INPUT_DIR, f)
        if os.path.isdir(f_path):
            # skip directories
            continue
        # check if it is an image
        test_img = cv2.imread(f_path)
        if test_img is not None:
            IMAGE_PATH_LIST.append(f_path)
        else:
            # test if it is a video
            test_video_cap = cv2.VideoCapture(f_path)
            n_frames = int(test_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            test_video_cap.release()
            if n_frames > 0:
                VIDEO_PATH = f_path
                # CAPTURE = cv2.VideoCapture(f_path)
                # it is a video
                desired_img_format = '.jpg'
                video_frames_path, video_name_ext = convert_video_to_images(f_path, n_frames, desired_img_format)
                # add video frames to image list
                frame_list = sorted(os.listdir(video_frames_path), key = natural_sort_key)
                ## store information about those frames
                first_index = len(IMAGE_PATH_LIST)
                last_index = first_index + len(frame_list) # exclusive
                indexes_dict = {}
                indexes_dict['first_index'] = first_index
                indexes_dict['last_index'] = last_index
                VIDEO_NAME_DICT[video_name_ext] = indexes_dict
                IMAGE_PATH_LIST.extend((os.path.join(video_frames_path, frame) for frame in frame_list))
    last_img_index = len(IMAGE_PATH_LIST) - 1

    # create output directories
    if len(VIDEO_NAME_DICT) > 0:
        if not os.path.exists(TRACKER_DIR):
            os.makedirs(TRACKER_DIR)
    for ann_dir in annotation_formats:
        new_dir = os.path.join(OUTPUT_DIR, ann_dir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        for video_name_ext in VIDEO_NAME_DICT:
            new_video_dir = os.path.join(new_dir, video_name_ext)
            labeling_file_dir = new_video_dir + '/labeling_file.txt'
            if not os.path.exists(new_video_dir):
                os.makedirs(new_video_dir)
           
                open(labeling_file_dir, 'a').close()
            else:
                try:
                    copyfile(json_file_path, json_file_path[:-5]+'_backup.json')
                except Exception:
                    pass
                first_element = IMAGE_PATH_LIST[0]
                ending = IMAGE_PATH_LIST[0].split('_')[-1]
                root_name = first_element.replace(ending,'')
                read_darklabel_file(labeling_file_dir,root_name)
                # for img_path in IMAGE_PATH_LIST:
                #     labeling_file[img_path] = read_darklabel_file(labeling_file_dir,img_path)
                set_max_anchor()



    # # load class list
    # with open('class_list.txt') as f:
    #     CLASS_LIST = list(nonblank_lines(f))
    # #print(CLASS_LIST)
    # last_class_index = len(CLASS_LIST) - 1

    # Make the class colors the same each session
    # The colors are in BGR order because we're using OpenCV
    class_rgb = [
        (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0),
        (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128)]
    class_rgb = np.array(class_rgb)
    # If there are still more classes, add new colors randomly
    num_colors_missing = len(CLASS_LIST) - len(class_rgb)
    if num_colors_missing > 0:
        more_colors = np.random.randint(0, 255+1, size=(num_colors_missing, 3))
        class_rgb = np.vstack([class_rgb, more_colors])

    # create window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, 1000, 700)
    cv2.setMouseCallback(WINDOW_NAME, mouse_listener)

    # selected image
    cv2.createTrackbar(TRACKBAR_IMG, WINDOW_NAME, 0, last_img_index, set_img_index)

    # selected class
    if last_class_index != 0:
        cv2.createTrackbar(TRACKBAR_CLASS, WINDOW_NAME, 0, last_class_index, set_class_index)

    # initialize
    set_img_index(0)
    edges_on = False

    display_text('Welcome!\n Press [h] for help.', 4000)

    # loop
    img_path = IMAGE_PATH_LIST[img_index]
    is_from_video, CURR_VIDEO_NAME = is_frame_from_video(img_path)
    object_list=img_objects[:]
    if is_from_video:
        json_file_path = '{}.json'.format(os.path.join(TRACKER_DIR, CURR_VIDEO_NAME))
        file_exists, current_data = get_json_file_data(json_file_path)
    while True:
        color = class_rgb[class_index].tolist()
        # clone the img
        tmp_img = img.copy()
        height, width = tmp_img.shape[:2]
        if edges_on == True:
            # draw edges
            tmp_img = draw_edges(tmp_img)
        # draw vertical and horizontal guide lines
        draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
        # write selected class
        class_name = CLASS_LIST[class_index]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        margin = 3
        text_width, text_height = cv2.getTextSize(class_name+str(curr_anchor_id), font, font_scale, LINE_THICKNESS)[0]
        tmp_img = cv2.rectangle(tmp_img, (mouse_x + LINE_THICKNESS, mouse_y - LINE_THICKNESS), (mouse_x + text_width + margin, mouse_y - text_height - margin), complement_bgr(color), -1)
        tmp_img = cv2.putText(tmp_img, class_name+str(curr_anchor_id), (mouse_x + margin, mouse_y - margin), font, font_scale, color, LINE_THICKNESS, cv2.LINE_AA)
        # get annotation paths

        img_path = IMAGE_PATH_LIST[img_index]
        annotation_paths = get_annotation_paths(img_path, annotation_formats)

        object_list = img_objects[:]
        for o in object_list:
            x1_c, y1_c, x2_c, y2_c=get_tracked_icon(*o[1:5])
            tmp_img = draw_tracked_icon_grey(tmp_img,x1_c, y1_c, x2_c, y2_c)
        object_list = remove_already_tracked_objects(object_list, img_path, current_data)
        for o in object_list:
            x1_c, y1_c, x2_c, y2_c=get_tracked_icon(*o[1:5])
            tmp_img = draw_tracked_icon(tmp_img,x1_c, y1_c, x2_c, y2_c)

  

        if dragBBox.anchor_being_dragged is not None:
            dragBBox.handler_mouse_move(mouse_x, mouse_y)
        # draw already done bounding boxes
        # tmp_img = draw_bboxes_from_file(tmp_img, annotation_paths, width, height)
        tmp_img = draw_bboxes_from_dict(tmp_img,img_path, width, height)
        # if bounding box is selected add extra info
        if is_bbox_selected:
            tmp_img = draw_info_bb_selected(tmp_img)
        # if first click
        if point_1[0] != -1:
            # draw partial bbox
            cv2.rectangle(tmp_img, point_1, (mouse_x, mouse_y), color, LINE_THICKNESS)
            # if second click
            if point_2[0] != -1:
                # save the bounding box
                # save_bounding_box(annotation_paths, class_index, point_1, point_2, width, height)
                update_bounding_box(img_path,curr_anchor_id,int(class_index),point_1[0],point_1[1],point_2[0],point_2[1])
                
                # update_json_anchor_id()
                curr_anchor_id+=1
                save_darklabel_txt(labeling_file_dir)
            
                # reset the points
                point_1 = (-1, -1)
                point_2 = (-1, -1)

        cv2.imshow(WINDOW_NAME, tmp_img)
        pressed_key = cv2.waitKey(DELAY)

        if dragBBox.anchor_being_dragged is None:
            ''' Key Listeners START '''
            if pressed_key == ord('a') or pressed_key == ord('d'):
                # show previous image key listener
                if pressed_key == ord('a'):
                    img_index = decrease_index(img_index, last_img_index)
                # show next image key listener
                elif pressed_key == ord('d'):
                    img_index = increase_index(img_index, last_img_index)
                set_img_index(img_index)
                cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)
            elif pressed_key == ord('s') or pressed_key == ord('w'):
                # change down current class key listener
                if pressed_key == ord('s'):
                    class_index = decrease_index(class_index, last_class_index)
                # change up current class key listener
                elif pressed_key == ord('w'):
                    class_index = increase_index(class_index, last_class_index)
                draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
                set_class_index(class_index)
                cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, int(class_index))
                if is_bbox_selected:
                    if selected_bbox > len(img_objects):
                        continue
                    obj_to_edit = img_objects[selected_bbox]
                    edit_bbox(obj_to_edit, 'change_class:{}'.format(class_index))
            # help key listener
            elif pressed_key == ord('h'):
                text = ('[e] to show edges;\n'
                        '[q] to quit;\n'
                        '[a] or [d] to change Image;\n'
                        '[w] or [s] to change Class.\n'
                        )
                display_text(text, 5000)
            # show edges key listener
            # elif pressed_key == ord('b'):
            #     # merge trackids of consectutive frames (this frame with prev)
            #     current_img_path = IMAGE_PATH_LIST[img_index]
              
            #     is_from_video, video_name = is_frame_from_video(current_img_path)
            #     if is_from_video:
            #         # get json file corresponding to that video
            #         json_file_path = '{}.json'.format(os.path.join(TRACKER_DIR, video_name))
            #         file_exists, json_file_data = get_json_file_data(json_file_path)
            #         try:
            #             copyfile(json_file_path, json_file_path[:-5]+'_backup.json')
            #         except Exception:
            #             pass
            #         with open(json_file_path, 'w') as outfile:
            #             # json.dump(json_file_data, outfile, sort_keys=True, indent=4)
            #             json.dump(json_file_data, outfile, sort_keys=True, indent=4)
            #         save_darklabel_txt(labeling_file_dir)


            elif pressed_key == ord('m'):
                current_img_path = IMAGE_PATH_LIST[img_index]
              
                is_from_video, video_name = is_frame_from_video(current_img_path)
                if is_from_video:
                    # get json file corresponding to that video
                    json_file_path = '{}.json'.format(os.path.join(TRACKER_DIR, video_name))
                    file_exists, json_file_data = get_json_file_data(json_file_path)
                    try:
                        copyfile(json_file_path, json_file_path[:-5]+'_backup.json')
                    except Exception:
                        pass
                    with open(json_file_path, 'w') as outfile:
                        # json.dump(json_file_data, outfile, sort_keys=True, indent=4)
                        json.dump(json_file_data, outfile, sort_keys=True, indent=4)
                save_darklabel_txt(labeling_file_dir)

                # show_curr_tracked =  not show_curr_tracked
                img_path = IMAGE_PATH_LIST[img_index]
                is_from_video, CURR_VIDEO_NAME = is_frame_from_video(img_path)
                object_list=img_objects[:]
                if is_from_video:
                    json_file_path = '{}.json'.format(os.path.join(TRACKER_DIR, CURR_VIDEO_NAME))
                    file_exists, current_data = get_json_file_data(json_file_path)

                    # if file_exists:
                    #     object_list = remove_already_tracked_objects(object_list, img_path, current_data)
                    #     for o in object_list:
                    #         x1_c, y1_c, x2_c, y2_c=get_tracked_icon(o[0],o[1],o[2],o[3])
                    #         tmp_img = draw_tracked_icon(tmp_img,x1_c, y1_c, x2_c, y2_c)

            elif pressed_key == ord('e'):
                if edges_on == True:
                    edges_on = False
                    display_text('Edges turned OFF!', 1000)
                else:
                    edges_on = True
                    display_text('Edges turned ON!', 1000)
            elif pressed_key == ord('p'):
                # check if the image is a frame from a video
                is_from_video, video_name = is_frame_from_video(img_path)
                if is_from_video:
                    # get list of objects associated to that frame
                    object_list = img_objects[:]
                    # remove the objects in that frame that are already in the `.json` file
                    json_file_path = '{}.json'.format(os.path.join(TRACKER_DIR, video_name))
                    file_exists, json_file_data = get_json_file_data(json_file_path)
                    if file_exists:
                        object_list = remove_already_tracked_objects(object_list, img_path, json_file_data)
                    if len(object_list) > 0:
                        # get list of frames following this image
                        next_frame_path_list = get_next_frame_path_list(video_name, img_path)
                        # initial frame
                        init_frame = img.copy()
                        label_tracker = LabelTracker(TRACKER_TYPE, init_frame, next_frame_path_list)
                        backup_tracker_frame = img_index
                        for obj in object_list:
                            img_index = backup_tracker_frame
                            set_img_index(img_index)
                            cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)

                            class_index = obj[-2]
                            color = class_rgb[class_index].tolist()
                            label_tracker.start_tracker(json_file_data, json_file_path, img_path, obj, color, annotation_formats)
            elif pressed_key == ord('o'):
                im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # boxes, confidences, classIds =  detector.detect(im_rgb)
                image_size = 1024
                height, width = im_rgb.shape[:2]
                image = im_rgb.astype(np.float32)
                # image = im_rgb.astype(np.float32) / 255
                # image[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                # image[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                # image[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
                if height > width:
                    scale = image_size / height
                    resized_height = image_size
                    resized_width = int(width * scale)
                else:
                    scale = image_size / width
                    resized_height = int(height * scale)
                    resized_width = image_size

                image = cv2.resize(image, (resized_width, resized_height))

                new_image = np.zeros((image_size, image_size, 3))
                new_image[0:resized_height, 0:resized_width] = image
                new_image = np.transpose(new_image, (2, 0, 1))
                new_image = new_image[None, :, :, :]
                new_image = torch.Tensor(new_image)

                if torch.cuda.is_available():
                    new_image = new_image.cuda()
                with torch.no_grad():
                    y = detector([{'image': new_image.squeeze()}], is_training=False)[0]
                    confidences = y['instances'].get('scores')
                    classIds = y['instances'].get('pred_classes')
                    boxes = y['instances'].get('pred_boxes').tensor
                    # confidences, classIds, boxes = detector(new_image) # boxes are xmin ymin xmax ymax
                    boxes /= scale
                boxes[:,2]=boxes[:,2]-boxes[:,0] # we need x y w h
                boxes[:,3]=boxes[:,3]-boxes[:,1]
                boxes=boxes[confidences>0.4].cpu() 
                classIds=classIds[confidences>0.4].cpu()
                confidences=confidences[confidences>0.4].cpu()
                if  len(boxes)>0:
                    # object_list=img_objects[:]
                    for box,class_index in zip(boxes,classIds):
                        update_bounding_box(img_path,curr_anchor_id,int(class_index),int(box[0]),int(box[1]),(int(box[0]) + int(box[2])),(int(box[1]) + int(box[3])))
                        curr_anchor_id+=1
                save_darklabel_txt(labeling_file_dir)

            elif pressed_key == ord(' '):
                current_img_path = IMAGE_PATH_LIST[img_index]
              
                is_from_video, video_name = is_frame_from_video(current_img_path)
                if is_from_video:
                    # get json file corresponding to that video
                    json_file_path = '{}.json'.format(os.path.join(TRACKER_DIR, video_name))
                    file_exists, json_file_data = get_json_file_data(json_file_path)
                    try:
                        copyfile(json_file_path, json_file_path[:-5]+'_backup.json')
                    except Exception:
                        pass
                    with open(json_file_path, 'w') as outfile:
                        # json.dump(json_file_data, outfile, sort_keys=True, indent=4)
                        json.dump(json_file_data, outfile, sort_keys=True, indent=4)
                save_darklabel_txt(labeling_file_dir)
    
            # quit key listener
            elif pressed_key == ord('q'):
                current_img_path = IMAGE_PATH_LIST[img_index]
              
                is_from_video, video_name = is_frame_from_video(current_img_path)
                if is_from_video:
                    # get json file corresponding to that video
                    json_file_path = '{}.json'.format(os.path.join(TRACKER_DIR, video_name))
                    file_exists, json_file_data = get_json_file_data(json_file_path)
                    try:
                        copyfile(json_file_path, json_file_path[:-5]+'_backup.json')
                    except Exception:
                        pass
                    with open(json_file_path, 'w') as outfile:
                        # json.dump(json_file_data, outfile, sort_keys=True, indent=4)
                        json.dump(json_file_data, outfile, sort_keys=True, indent=4)
                save_darklabel_txt(labeling_file_dir)
                
                break
            ''' Key Listeners END '''

        if WITH_QT:
            # if window gets closed then quit
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
