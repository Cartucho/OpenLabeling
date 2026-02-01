"""
Object tracking module for OpenLabeling
"""
import cv2
import json
from typing import List, Tuple, Dict, Any, Optional
from .utils import yolo_format


class LabelTracker:
    """Special thanks to Rafael Caballero Gonzalez"""
    
    def __init__(self, tracker_type: str, init_frame, next_frame_path_list: List[str], config):
        tracker_types = ['CSRT', 'KCF','MOSSE', 'MIL', 'BOOSTING', 'MEDIANFLOW', 'TLD', 'GOTURN', 'DASIAMRPN']
        ''' Recommended tracker_type:
              KCF -> KCF is usually very good (minimum OpenCV 3.1.0)
              CSRT -> More accurate than KCF but slightly slower (minimum OpenCV 3.4.2)
              MOSSE -> Less accurate than KCF but very fast (minimum OpenCV 3.4.1)
        '''
        self.tracker_type = tracker_type
        self.config = config
        # Extract the OpenCV version info, e.g.:
        # OpenCV 3.3.4 -> [major_ver].[minor_ver].[subminor_ver]
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        self.major_ver = major_ver
        self.minor_ver = minor_ver
        self.subminor_ver = subminor_ver
        
        # TODO: press ESC to stop the tracking process
        
        # -- TODO: remove this if I assume OpenCV version > 3.4.0
        if tracker_type == tracker_types[0] or tracker_type == tracker_types[2]:
            if int(self.major_ver == 3) and int(self.minor_ver) < 4:
                self.tracker_type = tracker_types[1]  # Use KCF instead of CSRT or MOSSE
        # --
        self.init_frame = init_frame
        self.next_frame_path_list = next_frame_path_list

        self.img_h, self.img_w = init_frame.shape[:2]

    def call_tracker_constructor(self, tracker_type: str):
        if tracker_type == 'DASIAMRPN':
            # Import dasiamrpn only when needed
            try:
                from ..main.dasiamrpn import dasiamrpn
                tracker = dasiamrpn()
            except ImportError:
                print("DASIAMRPN not available. Using KCF instead.")
                tracker = cv2.TrackerKCF_create()
        else:
            # -- TODO: remove this if I assume OpenCV version > 3.4.0
            if int(self.major_ver == 3) and int(self.minor_ver) < 3:
                # tracker = cv2.Tracker_create(tracker_type)
                pass
            # --
            else:
                try:
                    tracker = cv2.TrackerKCF_create()
                except AttributeError as error:
                    print(error)
                    print('\nMake sure that OpenCV contribute is installed: opencv-contrib-python\n')
                    return None
                    
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

    def start_tracker(self, json_file_data: Dict[str, Any], json_file_path: str, img_path: str, 
                     obj: List, color: Tuple[int, int, int], annotation_formats: Dict[str, str], 
                     bbox_handler, class_index: int, class_rgb):
        tracker = self.call_tracker_constructor(self.tracker_type)
        if tracker is None:
            return
            
        anchor_id = json_file_data['n_anchor_ids']
        frame_data_dict = json_file_data['frame_data_dict']

        pred_counter = 0
        frame_data_dict = self._json_file_add_object(frame_data_dict, img_path, anchor_id, pred_counter, obj)
        # tracker bbox format: xmin, ymin, w, h
        xmin, ymin, xmax, ymax = obj[1:5]
        initial_bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
        tracker.init(self.init_frame, initial_bbox)
        for frame_path in self.next_frame_path_list:
            next_image = cv2.imread(frame_path)
            # get the new bbox prediction of the object
            success, bbox = tracker.update(next_image.copy())
            if pred_counter >= self.config.n_frames:
                success = False
            if success:
                pred_counter += 1
                xmin, ymin, w, h = map(int, bbox)
                xmax = xmin + w
                ymax = ymin + h
                new_obj = [class_index, xmin, ymin, xmax, ymax]
                frame_data_dict = self._json_file_add_object(frame_data_dict, frame_path, anchor_id, pred_counter, new_obj)
                cv2.rectangle(next_image, (xmin, ymin), (xmax, ymax), color, self.config.thickness)
                # save prediction
                annotation_paths = bbox_handler.get_annotation_paths(frame_path, annotation_formats)
                
                # Save bounding box to annotation files
                for ann_path in annotation_paths:
                    if '.txt' in ann_path:
                        line = yolo_format(class_index, (xmin, ymin), (xmax, ymax), self.img_w, self.img_h)
                        self._append_bb(ann_path, line, '.txt')
                    elif '.xml' in ann_path:
                        line = self._voc_format_with_tuple(class_index, (xmin, ymin), (xmax, ymax))
                        self._append_bb(ann_path, line, '.xml')
                        
                # show prediction
                cv2.imshow(self.config.window_name, next_image)
                pressed_key = cv2.waitKey(self.config.delay)
                
                # Check if user wants to quit
                if pressed_key == ord('q'):
                    break
            else:
                break

        json_file_data.update({'n_anchor_ids': (anchor_id + 1)})
        # save the updated data
        with open(json_file_path, 'w') as outfile:
            json.dump(json_file_data, outfile, sort_keys=True, indent=4)
    
    def _json_file_add_object(self, frame_data_dict: Dict[str, Any], img_path: str, anchor_id: int, pred_counter: int, obj: List) -> Dict[str, Any]:
        import xml.etree.cElementTree as ET
        
        object_list = self._get_json_file_object_list(img_path, frame_data_dict)
        class_index, xmin, ymin, xmax, ymax = obj

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
    
    def _get_json_file_object_list(self, img_path: str, frame_data_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        object_list = []
        if img_path in frame_data_dict:
            object_list = frame_data_dict[img_path]
        return object_list
    
    def _voc_format_with_tuple(self, class_index: int, point_1: Tuple[int, int], point_2: Tuple[int, int]) -> Tuple[str, str, str, str, str]:
        """
        Convert bounding box coordinates to VOC format using class index
        Order: class_name xmin ymin xmax ymax
        """
        from .utils import voc_format
        class_name = self.config.class_list[class_index]
        return voc_format(class_name, point_1, point_2)
    
    def _append_bb(self, ann_path: str, line, extension: str):
        """Append a bounding box to an annotation file"""
        import xml.etree.cElementTree as ET
        
        if '.txt' in extension:
            with open(ann_path, 'a') as myfile:
                myfile.write(line + '\n')  # append line
        elif '.xml' in extension:
            class_name, xmin, ymin, xmax, ymax = line

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

            xml_str = ET.tostring(annotation)
            # Use the write_xml function from bbox_handler if available
            try:
                from .bbox_handler import BoundingBoxHandler
                handler = BoundingBoxHandler(self.config)
                handler.write_xml(xml_str, ann_path)
            except:
                # Fallback to basic writing
                with open(ann_path, 'wb') as temp_xml:
                    temp_xml.write(xml_str)