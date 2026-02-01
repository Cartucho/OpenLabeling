"""
Bounding box handling module for OpenLabeling
"""
import os
import json
from typing import List, Tuple, Dict, Any, Optional
from lxml import etree
import xml.etree.cElementTree as ET
import cv2
import numpy as np
from .config import Config
from .utils import get_anchors_rectangles, point_in_rect, get_bbox_area


class DragBoundingBox:
    """
    Class to deal with bbox resizing
    Anchors: 
        LT -- MT -- RT
        |            |
        LM          RM  
        |            |
        LB -- MB -- RB
    """

    def __init__(self, line_thickness: int):
        # Size of resizing anchors (depends on LINE_THICKNESS)
        self.sra = line_thickness * 2
        # Object being dragged
        self.selected_object = None
        # Flag indicating which resizing-anchor is dragged
        self.anchor_being_dragged = None

    def check_point_inside_resizing_anchors(self, e_x: int, e_y: int, obj: List) -> None:
        """
        Check if a current mouse position is inside one of the resizing anchors of a bbox
        """
        _, x_left, y_top, x_right, y_bottom = obj
        # first check if inside the bbox region (to avoid making 8 comparisons per object)
        if point_in_rect(e_x, e_y,
                        x_left - self.sra,
                        y_top - self.sra,
                        x_right + self.sra,
                        y_bottom + self.sra):

            anchor_dict = get_anchors_rectangles(x_left, y_top, x_right, y_bottom, self.sra)
            for anchor_key in anchor_dict:
                r_x_left, r_y_top, r_x_right, r_y_bottom = anchor_dict[anchor_key]
                if point_in_rect(e_x, e_y, r_x_left, r_y_top, r_x_right, r_y_bottom):
                    self.anchor_being_dragged = anchor_key
                    break

    def handler_left_mouse_down(self, e_x: int, e_y: int, obj: List) -> None:
        """
        Select an object if one presses a resizing anchor
        """
        self.check_point_inside_resizing_anchors(e_x, e_y, obj)
        if self.anchor_being_dragged is not None:
            self.selected_object = obj

    def handler_mouse_move(self, e_x: int, e_y: int, edit_bbox_callback) -> None:
        """
        Handle mouse movement during dragging
        """
        if self.selected_object is not None:
            class_name, x_left, y_top, x_right, y_bottom = self.selected_object

            # Do not allow the bbox to flip upside down (given a margin)
            margin = 3 * self.sra
            change_was_made = False

            if self.anchor_being_dragged[0] == "L":
                # left anchors (LT, LM, LB)
                if e_x < x_right - margin:
                    x_left = e_x
                    change_was_made = True
            elif self.anchor_being_dragged[0] == "R":
                # right anchors (RT, RM, RB)
                if e_x > x_left + margin:
                    x_right = e_x
                    change_was_made = True

            if self.anchor_being_dragged[1] == "T":
                # top anchors (LT, RT, MT)
                if e_y < y_bottom - margin:
                    y_top = e_y
                    change_was_made = True
            elif self.anchor_being_dragged[1] == "B":
                # bottom anchors (LB, RB, MB)
                if e_y > y_top + margin:
                    y_bottom = e_y
                    change_was_made = True

            if change_was_made:
                action = "resize_bbox:{}:{}:{}:{}".format(x_left, y_top, x_right, y_bottom)
                edit_bbox_callback(self.selected_object, action)
                # update the selected bbox
                self.selected_object = [class_name, x_left, y_top, x_right, y_bottom]

    def handler_left_mouse_up(self, e_x: int, e_y: int) -> None:
        """
        Reset this class on mouse up
        """
        if self.selected_object is not None:
            self.selected_object = None
            self.anchor_being_dragged = None


class BoundingBoxHandler:
    """
    Handles bounding box operations including creation, modification, and deletion
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.tracker_dir = os.path.join(config.output_dir, '.tracker')

    def get_xml_object_data(self, obj: ET.Element) -> List:
        """
        Extract object data from XML element
        """
        class_name = obj.find('name').text
        class_index = self.config.class_list.index(class_name)
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        return [class_name, class_index, xmin, ymin, xmax, ymax]

    def get_txt_object_data(self, obj: str, img_width: int, img_height: int) -> List:
        """
        Extract object data from YOLO text format
        """
        class_id, center_x, center_y, bbox_width, bbox_height = obj.split()
        bbox_width = float(bbox_width)
        bbox_height = float(bbox_height)
        center_x = float(center_x)
        center_y = float(center_y)

        class_index = int(class_id)
        class_name = self.config.class_list[class_index]
        xmin = int(img_width * center_x - img_width * bbox_width/2.0)
        xmax = int(img_width * center_x + img_width * bbox_width/2.0)
        ymin = int(img_height * center_y - img_height * bbox_height/2.0)
        ymax = int(img_height * center_y + img_height * bbox_height/2.0)
        return [class_name, class_index, xmin, ymin, xmax, ymax]

    def draw_bbox_anchors(self, tmp_img: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int, color: Tuple[int, int, int]) -> np.ndarray:
        """
        Draw resizing anchors on a bounding box
        """
        anchor_dict = get_anchors_rectangles(xmin, ymin, xmax, ymax, self.config.resizing_anchor_size)
        for anchor_key in anchor_dict:
            x1, y1, x2, y2 = anchor_dict[anchor_key]
            cv2.rectangle(tmp_img, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
        return tmp_img

    def draw_bboxes_from_file(self, tmp_img: np.ndarray, annotation_paths: List[str], width: int, height: int, 
                              img_objects: List, is_bbox_selected: bool, selected_bbox: int, class_rgb: np.ndarray) -> np.ndarray:
        """
        Draw bounding boxes from annotation files onto the image
        """
        img_objects.clear()
        ann_path = None
        if self.config.draw_from_pascal:
            # Drawing bounding boxes from the PASCAL files
            ann_path = next(path for path in annotation_paths if 'PASCAL_VOC' in path)
        else:
            # Drawing bounding boxes from the YOLO files
            ann_path = next(path for path in annotation_paths if 'YOLO_darknet' in path)
        
        if os.path.isfile(ann_path):
            if self.config.draw_from_pascal:
                tree = ET.parse(ann_path)
                annotation = tree.getroot()
                for idx, obj in enumerate(annotation.findall('object')):
                    class_name, class_index, xmin, ymin, xmax, ymax = self.get_xml_object_data(obj)
                    img_objects.append([class_index, xmin, ymin, xmax, ymax])
                    color = class_rgb[class_index].tolist()
                    # draw bbox
                    cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), color, self.config.thickness)
                    # draw resizing anchors if the object is selected
                    if is_bbox_selected:
                        if idx == selected_bbox:
                            tmp_img = self.draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, color)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(tmp_img, class_name, (xmin, ymin - 5), font, 0.6, color, self.config.thickness, cv2.LINE_AA)
            else:
                # Draw from YOLO
                with open(ann_path) as fp:
                    for idx, line in enumerate(fp):
                        obj = line
                        class_name, class_index, xmin, ymin, xmax, ymax = self.get_txt_object_data(obj, width, height)
                        img_objects.append([class_index, xmin, ymin, xmax, ymax])
                        color = class_rgb[class_index].tolist()
                        # draw bbox
                        cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), color, self.config.thickness)
                        # draw resizing anchors if the object is selected
                        if is_bbox_selected:
                            if idx == selected_bbox:
                                tmp_img = self.draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, color)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(tmp_img, class_name, (xmin, ymin - 5), font, 0.6, color, self.config.thickness, cv2.LINE_AA)
        return tmp_img

    def set_selected_bbox(self, img_objects: List, mouse_x: int, mouse_y: int, drag_bbox: DragBoundingBox) -> Tuple[bool, int]:
        """
        Set the selected bounding box based on mouse position
        """
        smallest_area = -1
        is_bbox_selected = False
        selected_bbox = -1
        # if clicked inside multiple bboxes selects the smallest one
        for idx, obj in enumerate(img_objects):
            ind, x1, y1, x2, y2 = obj
            x1 = x1 - drag_bbox.sra
            y1 = y1 - drag_bbox.sra
            x2 = x2 + drag_bbox.sra
            y2 = y2 + drag_bbox.sra
            if point_in_rect(mouse_x, mouse_y, x1, y1, x2, y2):
                is_bbox_selected = True
                tmp_area = get_bbox_area(x1, y1, x2, y2)
                if tmp_area < smallest_area or smallest_area == -1:
                    smallest_area = tmp_area
                    selected_bbox = idx
        return is_bbox_selected, selected_bbox

    def is_mouse_inside_delete_button(self, img_objects: List, selected_bbox: int, mouse_x: int, mouse_y: int) -> bool:
        """
        Check if mouse is inside the delete button of the selected bounding box
        """
        for idx, obj in enumerate(img_objects):
            if idx == selected_bbox:
                _ind, x1, y1, x2, y2 = obj
                x1_c, y1_c, x2_c, y2_c = self.get_close_icon(x1, y1, x2, y2)
                if point_in_rect(mouse_x, mouse_y, x1_c, y1_c, x2_c, y2_c):
                    return True
        return False

    def get_close_icon(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
        """
        Get coordinates for the close icon
        """
        percentage = 0.05
        height = -1
        while height < 15 and percentage < 1.0:
            height = int((y2 - y1) * percentage)
            percentage += 0.1
        return (x2 - height), y1, x2, (y1 + height)

    def edit_bbox(self, obj_to_edit: List, action: str, image_path_list: List[str], img_index: int, 
                  annotation_formats: Dict[str, str], width: int, height: int, video_name_dict: Dict) -> None:
        """
        Edit a bounding box based on the action
        """
        if 'change_class' in action:
            new_class_index = int(action.split(':')[1])
        elif 'resize_bbox' in action:
            new_x_left = max(0, int(action.split(':')[1]))
            new_y_top = max(0, int(action.split(':')[2]))
            new_x_right = min(width, int(action.split(':')[3]))
            new_y_bottom = min(height, int(action.split(':')[4]))

        # 1. initialize bboxes_to_edit_dict
        #    (we use a dict since a single label can be associated with multiple ones in videos)
        bboxes_to_edit_dict = {}
        current_img_path = image_path_list[img_index]
        bboxes_to_edit_dict[current_img_path] = obj_to_edit

        # 2. add elements to bboxes_to_edit_dict
        '''
            If the bbox is in the json file then it was used by the video Tracker, hence,
            we must also edit the next predicted bboxes associated to the same `anchor_id`.
        '''
        # if `current_img_path` is a frame from a video
        is_from_video, video_name = self.is_frame_from_video(current_img_path, video_name_dict)
        if is_from_video:
            # get json file corresponding to that video
            json_file_path = '{}.json'.format(os.path.join(self.tracker_dir, video_name))
            file_exists, json_file_data = self.get_json_file_data(json_file_path)
            # if json file exists
            if file_exists:
                # match obj_to_edit with the corresponding json object
                frame_data_dict = json_file_data['frame_data_dict']
                json_object_list = self.get_json_file_object_list(current_img_path, frame_data_dict)
                obj_matched = self.get_json_object_dict(obj_to_edit, json_object_list)
                # if match found
                if obj_matched is not None:
                    # get this object's anchor_id
                    anchor_id = obj_matched['anchor_id']

                    frame_path_list = self.get_next_frame_path_list(video_name, current_img_path, image_path_list, video_name_dict)
                    frame_path_list.insert(0, current_img_path)

                    if 'change_class' in action:
                        # add also the previous frames
                        prev_path_list = self.get_prev_frame_path_list(video_name, current_img_path, image_path_list, video_name_dict)
                        frame_path_list = prev_path_list + frame_path_list

                    # update json file if contain the same anchor_id
                    for frame_path in frame_path_list:
                        json_object_list = self.get_json_file_object_list(frame_path, frame_data_dict)
                        json_obj = self.get_json_file_object_by_id(json_object_list, anchor_id)
                        if json_obj is not None:
                            bboxes_to_edit_dict[frame_path] = [
                                json_obj['class_index'],
                                json_obj['bbox']['xmin'],
                                json_obj['bbox']['ymin'],
                                json_obj['bbox']['xmax'],
                                json_obj['bbox']['ymax']
                            ]
                            # edit json file
                            if 'delete' in action:
                                json_object_list.remove(json_obj)
                            elif 'change_class' in action:
                                json_obj['class_index'] = new_class_index
                            elif 'resize_bbox' in action:
                                json_obj['bbox']['xmin'] = new_x_left
                                json_obj['bbox']['ymin'] = new_y_top
                                json_obj['bbox']['xmax'] = new_x_right
                                json_obj['bbox']['ymax'] = new_y_bottom
                        else:
                            break

                    # save the edited data
                    with open(json_file_path, 'w') as outfile:
                        json.dump(json_file_data, outfile, sort_keys=True, indent=4)

        # 3. loop through bboxes_to_edit_dict and edit the corresponding annotation files
        for path in bboxes_to_edit_dict:
            obj_to_edit = bboxes_to_edit_dict[path]
            class_index, xmin, ymin, xmax, ymax = map(int, obj_to_edit)

            for ann_path in self.get_annotation_paths(path, annotation_formats):
                if '.txt' in ann_path:
                    # edit YOLO file
                    with open(ann_path, 'r') as old_file:
                        lines = old_file.readlines()

                    from .utils import yolo_format
                    yolo_line = yolo_format(class_index, (xmin, ymin), (xmax, ymax), width, height)
                    ind = self.find_index(obj_to_edit, img_objects)
                    i = 0

                    with open(ann_path, 'w') as new_file:
                        for line in lines:

                            if i != ind:
                               new_file.write(line)

                            elif 'change_class' in action:
                                new_yolo_line = yolo_format(new_class_index, (xmin, ymin), (xmax, ymax), width, height)
                                new_file.write(new_yolo_line + '\n')
                            elif 'resize_bbox' in action:
                                new_yolo_line = yolo_format(class_index, (new_x_left, new_y_top), (new_x_right, new_y_bottom), width, height)
                                new_file.write(new_yolo_line + '\n')

                            i = i + 1

                elif '.xml' in ann_path:
                    # edit PASCAL VOC file
                    tree = ET.parse(ann_path)
                    annotation = tree.getroot()
                    for obj in annotation.findall('object'):
                        class_name_xml, class_index_xml, xmin_xml, ymin_xml, xmax_xml, ymax_xml = self.get_xml_object_data(obj)
                        if ( class_index == class_index_xml and
                                         xmin == xmin_xml and
                                         ymin == ymin_xml and
                                         xmax == xmax_xml and
                                         ymax == ymax_xml ) :
                            if 'delete' in action:
                                annotation.remove(obj)
                            elif 'change_class' in action:
                                # edit object class name
                                object_class = obj.find('name')
                                object_class.text = self.config.class_list[new_class_index]
                            elif 'resize_bbox' in action:
                                object_bbox = obj.find('bndbox')
                                object_bbox.find('xmin').text = str(new_x_left)
                                object_bbox.find('ymin').text = str(new_y_top)
                                object_bbox.find('xmax').text = str(new_x_right)
                                object_bbox.find('ymax').text = str(new_y_bottom)
                            break

                    xml_str = ET.tostring(annotation)
                    self.write_xml(xml_str, ann_path)

    def find_index(self, obj_to_find: List, img_objects: List) -> int:
        """
        Find the index of an object in the img_objects list
        """
        for ind, list_elem in enumerate(img_objects):
            if list_elem == obj_to_find:
                return ind
        return -1

    def write_xml(self, xml_str: bytes, xml_path: str) -> None:
        """
        Write XML string to file with proper formatting
        """
        # remove blank text before prettifying the xml
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.fromstring(xml_str, parser)
        # prettify
        xml_str = etree.tostring(root, pretty_print=True)
        # save to file
        with open(xml_path, 'wb') as temp_xml:
            temp_xml.write(xml_str)

    def get_annotation_paths(self, img_path: str, annotation_formats: Dict[str, str]) -> List[str]:
        """
        Get annotation paths for an image
        """
        annotation_paths = []
        for ann_dir, ann_ext in annotation_formats.items():
            new_path = os.path.join(self.config.output_dir, ann_dir)
            new_path = os.path.join(new_path, os.path.basename(os.path.normpath(img_path)))
            pre_path, img_ext = os.path.splitext(new_path)
            new_path = new_path.replace(img_ext, ann_ext, 1)
            annotation_paths.append(new_path)
        return annotation_paths

    def is_frame_from_video(self, img_path: str, video_name_dict: Dict) -> Tuple[bool, str]:
        """
        Check if an image is a frame from a video
        """
        for video_name in video_name_dict:
            video_dir = os.path.join(self.config.input_dir, video_name)
            if os.path.dirname(img_path) == video_dir:
                # image belongs to a video
                return True, video_name
        return False, None

    def get_json_file_data(self, json_file_path: str) -> Tuple[bool, Dict]:
        """
        Load JSON file data
        """
        if os.path.isfile(json_file_path):
            with open(json_file_path) as f:
                data = json.load(f)
                return True, data
        else:
            return False, {'n_anchor_ids': 0, 'frame_data_dict': {}}

    def get_prev_frame_path_list(self, video_name: str, img_path: str, image_path_list: List[str], video_name_dict: Dict) -> List[str]:
        """
        Get previous frame paths for a video
        """
        first_index = video_name_dict[video_name]['first_index']
        img_index = image_path_list.index(img_path)
        return image_path_list[first_index:img_index]

    def get_next_frame_path_list(self, video_name: str, img_path: str, image_path_list: List[str], video_name_dict: Dict) -> List[str]:
        """
        Get next frame paths for a video
        """
        first_index = video_name_dict[video_name]['first_index']
        last_index = video_name_dict[video_name]['last_index']
        img_index = image_path_list.index(img_path)
        return image_path_list[(img_index + 1):last_index]

    def get_json_object_dict(self, obj: List, json_object_list: List[Dict]) -> Optional[Dict]:
        """
        Match an object with its JSON representation
        """
        if len(json_object_list) > 0:
            class_index, xmin, ymin, xmax, ymax = map(int, obj)
            for d in json_object_list:
                if ( d['class_index'] == class_index and
                     d['bbox']['xmin'] == xmin and
                     d['bbox']['ymin'] == ymin and
                     d['bbox']['xmax'] == xmax and
                     d['bbox']['ymax'] == ymax ) :
                    return d
        return None

    def get_json_file_object_by_id(self, json_object_list: List[Dict], anchor_id: int) -> Optional[Dict]:
        """
        Get a JSON object by its anchor ID
        """
        for obj_dict in json_object_list:
            if obj_dict['anchor_id'] == anchor_id:
                return obj_dict
        return None

    def get_json_file_object_list(self, img_path: str, frame_data_dict: Dict) -> List[Dict]:
        """
        Get JSON object list for a specific image path
        """
        object_list = []
        if img_path in frame_data_dict:
            object_list = frame_data_dict[img_path]
        return object_list