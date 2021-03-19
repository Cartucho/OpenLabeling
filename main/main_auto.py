#!/bin/python
import argparse
import json
import os
import re
import cv2
import numpy as np
from tqdm import tqdm
from shutil import copyfile
from datetime import datetime
 


import sys
sys.path.insert(0, "..")
# from object_detection.tf_object_detection import ObjectDetector
import configparser
from siammask import SiamMask
import torch
from centernet_better.train import CenterNetBetterModule


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


parser = argparse.ArgumentParser(description='Open-source image labeling tool')
parser.add_argument('-i', '--input_dir', default='input', type=str, help='Path to input directory')
parser.add_argument('-o', '--output_dir', default='output', type=str, help='Path to output directory')
parser.add_argument('-t', '--thickness', default='1', type=int, help='Bounding box and cross line thickness')
parser.add_argument('--detector', default='../object_detection/crow/epoch=46-step=17342.ckpt', type=str, help='Detector checkpoint dir')
parser.add_argument('--tracker', default='SiamMask', type=str, help="tracker_type being used: ['SiamMask']")

args = parser.parse_args()

class_index = 0
img_index = 0
is_last_frame = False
img = None
img_objects = []

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir

WINDOW_NAME = 'OpenLabeling'
TRACKBAR_IMG = 'Image'
TRACKBAR_CLASS = 'Class'

annotation_formats = {'Annotations' : '.txt'}
TRACKER_DIR = os.path.join(OUTPUT_DIR, '.tracker')

labeling_file = {} # dictionary which takes for every img entry: anchor_id,xmin,ymin,w,h,class_id,class_name
labeling_file_dir = None
curr_anchor_id = 0

# selected bounding box
prev_was_double_click = False
is_bbox_selected = False
selected_bbox = -1
LINE_THICKNESS = args.thickness

mouse_x = 0
mouse_y = 0
point_1 = (-1, -1)
point_2 = (-1, -1)


model = args.detector
if torch.cuda.is_available():
    detector = CenterNetBetterModule.load_from_checkpoint(model, pretrained_checkpoints_path=None)
    detector = detector.cuda()
else:
    detector = torch.load(model,map_location='cpu').module


def display_text(text, time):
    if WITH_QT:
        cv2.displayOverlay(WINDOW_NAME, text, time)
    else:
        print(text)

def set_img_index(x):
    global img_index, img
    global is_last_frame
    img_index = x
    if img_index < last_index:
        is_last_frame = False
    img_path = IMAGE_PATH_LIST[img_index]
    img = cv2.imread(img_path)
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
    global is_last_frame
    current_index += 1
    if current_index > last_index:
        current_index = 0
        is_last_frame = True
    return current_index


def draw_line(img, x, y, height, width, color):
    cv2.line(img, (x, 0), (x, height), color, LINE_THICKNESS)
    cv2.line(img, (0, y), (width, y), color, LINE_THICKNESS)





def append_bb(ann_path, line, extension):
    if '.txt' in extension:
        with open(ann_path, 'a') as myfile:
            myfile.write(line + '\n') # append line
   

        tree = ET.parse(ann_path)
        annotation = tree.getroot()

        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = class_name
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = xmin
        ET.SubElement(bbox, 'ymin').text = ymin
        ET.SubElement(bbox, 'xmax').text = xmax
        ET.SubElement(bbox, 'ymax').text = ymax


def draw_text(tmp_img, text, center, color, size):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(tmp_img, text, center, font, 0.6, color, size, cv2.LINE_AA)
    return tmp_img

def draw_bboxes_from_dict(tmp_img, img_path, width, height):
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
                anchor_id,xmin, ymin, w, h, class_index, class_name = obj
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


def get_close_icon(x1, y1, x2, y2):
    percentage = 0.05
    height = -1
    while height < 15 and percentage < 1.0:
        height = int((y2 - y1) * percentage)
        percentage += 0.1
    return (x2 - height), y1, x2, (y1 + height)


def draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c):
    red = (0,0,255)
    cv2.rectangle(tmp_img, (x1_c + 1, y1_c - 1), (x2_c, y2_c), red, -1)
    white = (255, 255, 255)
    cv2.line(tmp_img, (x1_c, y1_c), (x2_c, y2_c), white, 2)
    cv2.line(tmp_img, (x1_c, y2_c), (x2_c, y1_c), white, 2)
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


def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line


def get_annotation_paths(img_path, annotation_formats):
    annotation_paths = []
    for ann_dir, ann_ext in annotation_formats.items():
        new_path = os.path.join(OUTPUT_DIR, ann_dir)
        new_path = img_path.replace(INPUT_DIR, new_path, 1)
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
            data = json.load(f)
            return True, data
    else:
        return False, {'n_anchor_ids':0, 'frame_data_dict':{}}





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

def get_json_object_dict_percent(obj, json_object_list):
    if len(json_object_list) > 0:
        class_name = obj.pop(-1)
        anchor_idx,xmin, ymin, xmax, ymax,class_index = map(int, obj)
        obj.append(class_name)
        for d in json_object_list:
                    if ( d['class_index'] == class_index and
                         overlap_percent(obj,d))>0.1 :
                        return d
    return None


def remove_already_tracked_objects(object_list, img_path, json_file_data):
    frame_data_dict = json_file_data['frame_data_dict']
    json_object_list = get_json_file_object_list(img_path, frame_data_dict)
    # copy the list since we will be deleting elements without restarting the loop
    temp_object_list = object_list[:]
    for obj in temp_object_list:
        obj_dict = get_json_object_dict_percent(obj, json_object_list)
        if obj_dict is not None:
            object_list.remove(obj)
            # json_object_list.remove(obj_dict)
    return object_list

def overlap_percent(obj,json_obj2):
    obj1 = obj[1:5]
    obj2 = [json_obj2['bbox']['xmin'],json_obj2['bbox']['ymin'],json_obj2['bbox']['xmax'],json_obj2['bbox']['ymax']]
    SA=(obj1[0]-obj1[2])*(obj1[1]-obj1[3])
    SB=(obj2[0]-obj2[2])*(obj2[1]-obj2[3])
    SI=(max(0,-(max(obj1[0],obj2[0])-min(obj1[2],obj2[2])))) * (max(0,-(max(obj1[1],obj2[1])-min(obj1[3],obj2[3]))))
    # A_overlap=(max(obj1[0],obj2[0])-min(obj1[2],obj2[2]))*(max(obj1[1],obj2[1])-min(obj1[3],obj2[3]))
    # p_overlap = abs(A_overlap/(A1+A2-A_overlap))
    # p_overlap = abs(A_overlap/A1)
    SU = SA+SB-SI

    return SI/SU

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

def get_iou(obj1,obj2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

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


class Tracker:
    ''' Special thanks to Rafael Caballero Gonzalez '''
    # extract the OpenCV version info, e.g.:
    # OpenCV 3.3.4 -> [major_ver].[minor_ver].[subminor_ver]
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # TODO: press ESC to stop the tracking process

    def __init__(self, tracker_type, anchorId, classId):
        self.instance = self.call_tracker_constructor(tracker_type) # Tracker instance
        self.classId = classId # Id of object such as people, bicycle,...
        self.anchorId = anchorId # Id of tracker

    def call_tracker_constructor(self, tracker_type):
        # -- TODO: remove this if I assume OpenCV version > 3.4.0
        if int(self.major_ver == 3) and int(self.minor_ver) < 3:
            tracker = cv2.Tracker_create(tracker_type)
        # --
        else:
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

'''
\brief This is class to manage all the trackers
       This class will check if there is any "miss" tracker in current(init) frame:
            + If there is, stop TrackerManager
            + If there is not, using trackers to predict objects in the next frame. Continue until there is any "miss" tracker
'''
class TrackerManager:
    ''' Special thanks to Rafael Caballero Gonzalez '''
    # extract the OpenCV version info, e.g.:
    # OpenCV 3.3.4 -> [major_ver].[minor_ver].[subminor_ver]
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # TODO: press ESC to stop the tracking process

    def __init__(self, tracker_type, init_frame, next_frame_path_list):
        tracker_types = ['SiamMask']
        ''' Recomended tracker_type:
              KCF -> KCF is usually very good (minimum OpenCV 3.1.0)
              CSRT -> More accurate than KCF but slightly slower (minimum OpenCV 3.4.2)
              MOSSE -> Less accurate than KCF but very fast (minimum OpenCV 3.4.1)
        '''
        self.tracker_type = tracker_type
        # -- TODO: remove this if I assume OpenCV version > 3.4.0
        if tracker_type == 'SiamMask':
            self.tracker_type = 'SiamMask'
        elif tracker_type == tracker_types[0] or tracker_type == tracker_types[2]:
            if int(self.major_ver == 3) and int(self.minor_ver) < 4:
                self.tracker_type = tracker_types[1]  # Use KCF instead of CSRT or MOSSE
        # --
        self.init_frame = init_frame
        self.next_frame_path_list = next_frame_path_list
        self.trackers = []
        self.img_h, self.img_w = init_frame.shape[:2]
        self.tracker_data =[]


    '''
    \brief Init trackers
    '''
    def init_trackers(self, bboxes, classIds, json_file_data,json_file_path, img_path):
        global img_index
        global curr_anchor_id

        # anchor_id = json_file_data['n_anchor_ids']
        anchor_id = curr_anchor_id
        frame_data_dict = json_file_data['frame_data_dict']
        image = cv2.imread(img_path)
        # iii=0
        for box, classId in zip(bboxes, classIds):

            anchor_id = anchor_id + 1
            if self.tracker_type == 'SiamMask':
                
                initial_bbox = (box[0], box[1], box[2], box[3])
                tracker = SiamMask(anchorid=anchor_id, classid=classId,init_frame=self.init_frame,init_bbox=initial_bbox)
                data = [anchor_id,classId,self.init_frame,initial_bbox]
                tracker.init(self.init_frame,initial_bbox)
            else:
                tracker = Tracker(self.tracker_type, anchorId=anchor_id, classId=classId)
                initial_bbox = (box[0], box[1], box[2], box[3])
                tracker.instance.init(self.init_frame, initial_bbox)
            self.trackers.append(tracker)
            # self.tracker_data.append(data)

            # Initialize trackers on json files.
            pred_counter  = 0
            xmin, ymin, w, h = map(int, box)
            xmax = xmin + w
            ymax = ymin + h
            obj = [anchor_id, xmin, ymin, xmax, ymax,int(classId),CLASS_LIST[int(classId)]]
            frame_data_dict = json_file_add_object(frame_data_dict, img_path, anchor_id, pred_counter, obj)

            # Save prediction
            annotation_paths = get_annotation_paths(img_path, annotation_formats)
            # save_bounding_box(annotation_paths, int(classId), (xmin, ymin), (xmax, ymax), self.img_w, self.img_h)
            update_bounding_box(img_path,anchor_id,int(classId),xmin,ymin,xmax,ymax)
            


            #Draw
            # color = class_rgb[int(data[1])].tolist()
            color = class_rgb[int(tracker.classId)].tolist()
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, LINE_THICKNESS)

        img_index = increase_index(img_index, last_img_index)
        cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)
        cv2.imshow(WINDOW_NAME, image)
        pressed_key = cv2.waitKey(DELAY)

        curr_anchor_id= anchor_id
        json_file_data.update({'n_anchor_ids': (anchor_id + 1)})

        # save the updated data
        with open(json_file_path, 'w') as outfile:
            json.dump(json_file_data, outfile, sort_keys=True, indent=4)
        save_darklabel_txt(labeling_file_dir)


    def predict_next_frames(self,json_file_data,json_file_path):
        global img_index
        global curr_anchor_id

        # anchor_id = json_file_data['n_anchor_ids']
        anchor_id = curr_anchor_id
        frame_data_dict = json_file_data['frame_data_dict']

        pred_counter = 0
        
        for frame_path in self.next_frame_path_list:
            is_there_missed_tracker = False
            bboxes = []
            misses = []
            errors = []
            if len(self.trackers)==0:
                break
            # Check if there is any "miss" tracker
            for t,tracker in enumerate(self.trackers):
                next_image = cv2.imread(frame_path)
                if self.tracker_type=='SiamMask':
                    success, bbox  = tracker.update(next_image.copy())
                else:
                    success, bbox = tracker.instance.update(next_image.copy())
                bboxes.append(bbox)
                if not success:
                    is_there_missed_tracker = False
                    # break
                    misses.append(t)
            if len(misses)>0:
                # for m in misses:
                #     bboxes.pop(m)
                #     self.trackers.pop(m)
                bboxes = [i for j, i in enumerate(bboxes) if j not in misses]
                self.trackers = [i for j, i in enumerate(self.trackers) if j not in misses]
            # if there is no "miss" tracker, then save labelled objects into files and keep predict at the next frame
            if not is_there_missed_tracker:
                pred_counter += 1
                for i, tracker in enumerate(self.trackers):
                    box = bboxes[i]

                    xmin, ymin, w, h = map(int, box)
                    if xmin<0 or ymin<0:
                        errors.append(i)
                        continue
                    xmax = xmin + w
                    ymax = ymin + h
                    # obj = [int(tracker.classId), xmin, ymin, xmax, ymax]
                    obj = [int(tracker.anchorId), xmin, ymin, xmax, ymax,int(tracker.classId),CLASS_LIST[int(tracker.classId)]]
                    frame_data_dict = json_file_add_object(frame_data_dict, frame_path, int(tracker.anchorId), pred_counter, obj)

                    color = class_rgb[int(tracker.classId)].tolist()
                    cv2.rectangle(next_image, (xmin, ymin), (xmax, ymax), color, LINE_THICKNESS)

                    # save prediction
                    annotation_paths = get_annotation_paths(frame_path, annotation_formats)
                    # save_bounding_box(annotation_paths, int(tracker.classId), (xmin, ymin), (xmax, ymax), self.img_w, self.img_h)
                    update_bounding_box(frame_path,tracker.anchorId,int(tracker.classId),xmin,ymin,xmax,ymax)


                cv2.imshow(WINDOW_NAME, next_image)
                pressed_key = cv2.waitKey(DELAY)

                img_index = increase_index(img_index, last_img_index)

                cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)
            if len(errors)>0:
                # for e in errors:
                #     bboxes.pop(e)
                #     self.trackers.pop(e)
                bboxes = [i for j, i in enumerate(bboxes) if j not in errors]
                self.trackers = [i for j, i in enumerate(self.trackers) if j not in errors]
            # If there is "miss" traker, then break Tracker Manager.
            # Note:Ready to use "Object Detection" to detect object
            # else:
            #     break

        # json_file_data.update({'n_anchor_ids': (anchor_id + 1)})
        # curr_anchor_id +=1
        # save the updated data
        with open(json_file_path, 'w') as outfile:
            json.dump(json_file_data, outfile, sort_keys=True, indent=4)
        save_darklabel_txt(labeling_file_dir)

    
# change to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
                copyfile(labeling_file_dir, labeling_file_dir[:-4]+'_backup.txt')
                first_element = IMAGE_PATH_LIST[0]
                ending = IMAGE_PATH_LIST[0].split('_')[-1]
                root_name = first_element.replace(ending,'')
                read_darklabel_file(labeling_file_dir,root_name)
                set_max_anchor()

# create empty annotation files for each image, if it doesn't exist already
# for img_path in IMAGE_PATH_LIST:
#     # image info for the .xml file
#     test_img = cv2.imread(img_path)
#     abs_path = os.path.abspath(img_path)
#     folder_name = os.path.dirname(img_path)
#     image_name = os.path.basename(img_path)
#     img_height, img_width, depth = (str(number) for number in test_img.shape)

    
# load class list
with open('class_list.txt') as f:
    CLASS_LIST = list(nonblank_lines(f))
#print(CLASS_LIST)
last_class_index = len(CLASS_LIST) - 1

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
new_track = False
print(datetime.now())
while True:
    if img_index >= last_index:
        is_last_frame = True

    if is_last_frame and new_track:
        print("Reach to the last frame!!!!")
        new_track = False
        is_last_frame = False
        # set_img_index(0)
        # continue
        print(datetime.now())

        break
    elif is_last_frame:
        print("Reach to the last frame!!!!")
        print(datetime.now())

        break

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
    tmp_img = draw_text(tmp_img, CLASS_LIST[class_index], (mouse_x + 5, mouse_y - 5), color, LINE_THICKNESS)
    img_path = IMAGE_PATH_LIST[img_index]
    annotation_paths = get_annotation_paths(img_path, annotation_formats)

    # draw already done bounding boxes
    # tmp_img = draw_bboxes_from_file(tmp_img, annotation_paths, width, height)
    tmp_img = draw_bboxes_from_dict(tmp_img,img_path, width, height)


    """ Algorithms for automatically labeling!!!!!!!
    #1. Using object detection to find objects. Then save them into annotation path

    #2. If len(objects) == 0, then increase index and come back to Step 1
    #   If len(objects) >  0, then using init TrackerManager to automatically labeling for later frames:
    #   2.1 In TrackerManager, if detect there is any miss detection of any tracker, then comeback to Step 1.
    """

    # Using object detection to find PEOPLE
    print("Using Detector!!!!")
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

    # new_boxes = boxes[:,:]


    if not len(boxes):
        cv2.imshow(WINDOW_NAME, tmp_img)
        pressed_key = cv2.waitKey(DELAY)
        img_index = increase_index(img_index, last_img_index)
        cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)

    else:

        # object_list = []
        
        object_list=img_objects[:]
        for box,index in zip(boxes,classIds):
            overlap = False
            for img_box in object_list:
                if  index== img_box[-2] and overlap_percent_bbox([int(box[0]),int(box[1]), (int(box[0]) + int(box[2])),(int(box[1]) + int(box[3]))],[int(img_box[1]),int(img_box[2]), int(img_box[3]),int(img_box[4])])>0.5:
                    overlap=True
            if overlap == False:
                object_list.append([999,int(box[0]),int(box[1]), (int(box[0]) + int(box[2])),(int(box[1]) + int(box[3])),int(index),CLASS_LIST[index]])
        current_img_path = IMAGE_PATH_LIST[img_index]
        is_from_video, video_name = is_frame_from_video(current_img_path)

        # If it is video
        if is_from_video:
            next_frame_path_list = get_next_frame_path_list(video_name, current_img_path)
            json_file_path = '{}.json'.format(os.path.join(TRACKER_DIR, video_name))
            file_exists, json_file_data = get_json_file_data(json_file_path)
            init_frame = img.copy()

            # remove the objects in that frame that are already in the `.json` file
            json_file_path = '{}.json'.format(os.path.join(TRACKER_DIR, video_name))
            file_exists, json_file_data = get_json_file_data(json_file_path)
            if file_exists:
                object_list = remove_already_tracked_objects(object_list, img_path, json_file_data)
            if len(object_list) > 0:
                new_track =True
                return_to_index = img_index
                print("Using tracker!!!!")
                tracker_manager = TrackerManager(args.tracker, init_frame, next_frame_path_list)
                new_boxes_max = np.asarray([object_[1:5] for object_ in object_list])
                new_classIds = [object_[-2] for object_ in object_list]

                new_boxes_max[:,2]=new_boxes_max[:,2]-new_boxes_max[:,0] # we need x y w h
                new_boxes_max[:,3]=new_boxes_max[:,3]-new_boxes_max[:,1]
                new_boxes = new_boxes_max.tolist()
                # ne
                # I have to restructure this, instead of initation 100s of trackers and then predicting
                tracker_manager.init_trackers(new_boxes, new_classIds, json_file_data, json_file_path, current_img_path)
                tracker_manager.predict_next_frames(json_file_data,json_file_path)
                # json_file_data['n_anchor_ids']-=2
                set_img_index(return_to_index)
            else:
                img_index = increase_index(img_index, last_img_index)
                cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)

                cv2.imshow(WINDOW_NAME, tmp_img)
                pressed_key = cv2.waitKey(DELAY)


        else: # If it is image
            for box, classId in zip(boxes, classIds):
                # save_bounding_box(annotation_paths, classId, (int(box[0]), int(box[1])),
                #                   (int(box[0]) + int(box[2]), int(box[1]) + int(box[3])), width, height)
                update_bounding_box(img_path,curr_anchor_id,classId,(int(box[0]),int(box[1])), (int(box[0]) + int(box[2]),int(box[1]) + int(box[3])))
                curr_anchor_id+=1
                save_darklabel_txt(labeling_file_dir)

                xmin, ymin, w, h = map(int, box)
                xmax = xmin + w
                ymax = ymin + h
                color = class_rgb[int(classId)].tolist()
                cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), color, LINE_THICKNESS)

            img_index = increase_index(img_index, last_img_index)
            cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)

            cv2.imshow(WINDOW_NAME, tmp_img)
            pressed_key = cv2.waitKey(DELAY)





    if WITH_QT:
        # if window gets closed then quit
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

cv2.destroyAllWindows()

