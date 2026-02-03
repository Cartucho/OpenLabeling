"""
Main application module for OpenLabeling
"""
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional
from .config import Config
from .utils import (
    point_in_rect, draw_edges, draw_line, increase_index, decrease_index,
    natural_sort_key, nonblank_lines, yolo_format, voc_format,
    complement_bgr, get_close_icon
)
from .bbox_handler import DragBoundingBox, BoundingBoxHandler


class OpenLabelingApp:
    """
    Main application class for OpenLabeling
    """
    
    def __init__(self):
        self.config = Config()
        self.bbox_handler = BoundingBoxHandler(self.config)
        self.drag_bbox = DragBoundingBox(self.config.line_thickness)
        
        # Initialize global variables
        self.class_index = 0
        self.img_index = 0
        self.img = None
        self.img_objects = []
        
        # Mouse position
        self.mouse_x = 0
        self.mouse_y = 0
        self.point_1 = (-1, -1)
        self.point_2 = (-1, -1)
        
        # Bounding box selection
        self.prev_was_double_click = False
        self.is_bbox_selected = False
        self.selected_bbox = -1
        
        # Image and class indices
        self.image_path_list = []
        self.video_name_dict = {}
        self.class_list = []
        self.last_img_index = 0
        self.last_class_index = 0
        
        # Class colors
        self.class_rgb = None
        
        # UI state
        self.edges_on = False
        
        # Initialize the application
        self._initialize_app()
    
    def _initialize_app(self):
        """Initialize the application components"""
        # Change to the directory of this script
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Load images and videos
        self._load_images_and_videos()
        
        # Create output directories
        self._create_output_directories()
        
        # Create annotation files
        self._create_annotation_files()
        
        # Load class list
        self._load_class_list()
        
        # Setup class colors
        self._setup_class_colors()
        
        # Create UI window
        self._create_window()
    
    def _load_images_and_videos(self):
        """Load all images and videos from input directory"""
        for f in sorted(os.listdir(self.config.input_dir), key=natural_sort_key):
            f_path = os.path.join(self.config.input_dir, f)
            if os.path.isdir(f_path):
                # Skip directories
                continue
            
            # Check if it is an image
            test_img = cv2.imread(f_path)
            if test_img is not None:
                self.image_path_list.append(f_path)
            else:
                # Test if it is a video
                test_video_cap = cv2.VideoCapture(f_path)
                n_frames = int(test_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                test_video_cap.release()
                if n_frames > 0:
                    # It is a video
                    desired_img_format = '.jpg'
                    video_frames_path, video_name_ext = self._convert_video_to_images(f_path, n_frames, desired_img_format)
                    # Add video frames to image list
                    frame_list = sorted(os.listdir(video_frames_path), key=natural_sort_key)
                    # Store information about those frames
                    first_index = len(self.image_path_list)
                    last_index = first_index + len(frame_list)  # exclusive
                    indexes_dict = {
                        'first_index': first_index,
                        'last_index': last_index
                    }
                    self.video_name_dict[video_name_ext] = indexes_dict
                    self.image_path_list.extend((os.path.join(video_frames_path, frame) for frame in frame_list))
        
        self.last_img_index = len(self.image_path_list) - 1
    
    def _convert_video_to_images(self, video_path: str, n_frames: int, desired_img_format: str) -> Tuple[str, str]:
        """Convert video to individual frames"""
        # Create folder to store images (if video was not converted to images already)
        file_path, file_extension = os.path.splitext(video_path)
        # Append extension to avoid collision of videos with same name
        # e.g.: `video.mp4`, `video.avi` -> `video_mp4/`, `video_avi/`
        file_extension = file_extension.replace('.', '_')
        file_path += file_extension
        video_name_ext = os.path.basename(file_path)
        
        if not os.path.exists(file_path):
            print(' Converting video to individual frames...')
            cap = cv2.VideoCapture(video_path)
            os.makedirs(file_path)
            # Read the video
            for i in tqdm(range(n_frames)):
                if not cap.isOpened():
                    break
                # Capture frame-by-frame
                ret, frame = cap.read()
                if ret == True:
                    # Save each frame (we use this format to avoid repetitions)
                    frame_name = '{}_{}{}'.format(video_name_ext, i, desired_img_format)
                    frame_path = os.path.join(file_path, frame_name)
                    cv2.imwrite(frame_path, frame)
            # Release the video capture object
            cap.release()
        return file_path, video_name_ext
    
    def _create_output_directories(self):
        """Create output directories for annotations"""
        if len(self.video_name_dict) > 0:
            if not os.path.exists(self.bbox_handler.tracker_dir):
                os.makedirs(self.bbox_handler.tracker_dir)
        
        for ann_dir in self.config.annotation_formats:
            new_dir = os.path.join(self.config.output_dir, ann_dir)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            for video_name_ext in self.video_name_dict:
                new_video_dir = os.path.join(new_dir, video_name_ext)
                if not os.path.exists(new_video_dir):
                    os.makedirs(new_video_dir)
    
    def _create_annotation_files(self):
        """Create empty annotation files for each image if they don't exist"""
        for img_path in self.image_path_list:
            # Image info for the .xml file
            test_img = cv2.imread(img_path)
            abs_path = os.path.abspath(img_path)
            folder_name = os.path.dirname(img_path)
            image_name = os.path.basename(img_path)
            img_height, img_width, depth = (str(number) for number in test_img.shape)

            for ann_path in self.bbox_handler.get_annotation_paths(img_path, self.config.annotation_formats):
                if not os.path.isfile(ann_path):
                    if '.txt' in ann_path:
                        open(ann_path, 'a').close()
                    elif '.xml' in ann_path:
                        self._create_pascal_voc_xml(ann_path, abs_path, folder_name, image_name, img_height, img_width, depth)
    
    def _create_pascal_voc_xml(self, xml_path: str, abs_path: str, folder_name: str, image_name: str, img_height: str, img_width: str, depth: str):
        """Create a PASCAL VOC XML file"""
        import xml.etree.cElementTree as ET
        
        annotation = ET.Element('annotation')
        ET.SubElement(annotation, 'folder').text = folder_name
        ET.SubElement(annotation, 'filename').text = image_name
        ET.SubElement(annotation, 'path').text = abs_path
        source = ET.SubElement(annotation, 'source')
        ET.SubElement(source, 'database').text = 'Unknown'
        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = img_width
        ET.SubElement(size, 'height').text = img_height
        ET.SubElement(size, 'depth').text = depth
        ET.SubElement(annotation, 'segmented').text = '0'

        xml_str = ET.tostring(annotation)
        self.bbox_handler.write_xml(xml_str, xml_path)
    
    def _load_class_list(self):
        """Load the class list from file"""
        with open('class_list.txt') as f:
            self.class_list = list(nonblank_lines(f))
        self.last_class_index = len(self.class_list) - 1
        self.config.class_list = self.class_list  # Store in config for bbox_handler
    
    def _setup_class_colors(self):
        """Setup RGB colors for each class"""
        # Make the class colors the same each session
        # The colors are in BGR order because we're using OpenCV
        class_rgb = np.array([
            (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255),
            (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0),
            (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128)
        ])
        
        # If there are still more classes, add new colors randomly
        num_colors_missing = len(self.class_list) - len(class_rgb)
        if num_colors_missing > 0:
            more_colors = np.random.randint(0, 255+1, size=(num_colors_missing, 3))
            class_rgb = np.vstack([class_rgb, more_colors])
        
        self.class_rgb = class_rgb
    
    def _create_window(self):
        """Create the main application window"""
        cv2.namedWindow(self.config.window_name, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.config.window_name, 1000, 700)
        cv2.setMouseCallback(self.config.window_name, self._mouse_listener)
        
        # Selected image trackbar
        cv2.createTrackbar(self.config.trackbar_img, self.config.window_name, 0, self.last_img_index, self._set_img_index)
        
        # Selected class trackbar
        if self.last_class_index != 0:
            cv2.createTrackbar(self.config.trackbar_class, self.config.window_name, 0, self.last_class_index, self._set_class_index)
    
    def _set_img_index(self, x: int):
        """Set the current image index"""
        self.img_index = x
        img_path = self.image_path_list[self.img_index]
        self.img = cv2.imread(img_path)
        text = 'Showing image {}/{}, path: {}'.format(str(self.img_index), str(self.last_img_index), img_path)
        self._display_text(text, 1000)
    
    def _set_class_index(self, x: int):
        """Set the current class index"""
        self.class_index = x
        text = 'Selected class {}/{} -> {}'.format(str(self.class_index), str(self.last_class_index), self.class_list[self.class_index])
        self._display_text(text, 3000)
    
    def _display_text(self, text: str, time: int):
        """Display text overlay or print to console"""
        if self.config.with_qt:
            cv2.displayOverlay(self.config.window_name, text, time)
        else:
            print(text)
    
    def _mouse_listener(self, event, x, y, flags, param):
        """Handle mouse events"""
        set_class = True
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.prev_was_double_click = True
            #print('Double click')
            self.point_1 = (-1, -1)
            # if clicked inside a bounding box we set that bbox
            self._set_selected_bbox(set_class)
        # By AlexeyGy: delete via right-click
        elif event == cv2.EVENT_RBUTTONDOWN:
            set_class = False
            self._set_selected_bbox(set_class)
            if self.is_bbox_selected:
                obj_to_edit = self.img_objects[self.selected_bbox]
                self.bbox_handler.edit_bbox(
                    obj_to_edit, 'delete', 
                    self.image_path_list, self.img_index, 
                    self.config.annotation_formats, 
                    self.img.shape[1], self.img.shape[0], 
                    self.video_name_dict
                )
                self.is_bbox_selected = False
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.prev_was_double_click:
                #print('Finish double click')
                self.prev_was_double_click = False
            else:
                #print('Normal left click')

                # Check if mouse inside on of resizing anchors of the selected bbox
                if self.is_bbox_selected:
                    self.drag_bbox.handler_left_mouse_down(x, y, self.img_objects[self.selected_bbox])

                if self.drag_bbox.anchor_being_dragged is None:
                    if self.point_1[0] == -1:
                        if self.is_bbox_selected:
                            if self.bbox_handler.is_mouse_inside_delete_button(self.img_objects, self.selected_bbox, x, y):
                                self._set_selected_bbox(set_class)
                                obj_to_edit = self.img_objects[self.selected_bbox]
                                self.bbox_handler.edit_bbox(
                                    obj_to_edit, 'delete', 
                                    self.image_path_list, self.img_index, 
                                    self.config.annotation_formats, 
                                    self.img.shape[1], self.img.shape[0], 
                                    self.video_name_dict
                                )
                            self.is_bbox_selected = False
                        else:
                            # first click (start drawing a bounding box or delete an item)

                            self.point_1 = (x, y)
                    else:
                        # minimal size for bounding box to avoid errors
                        threshold = 5
                        if abs(x - self.point_1[0]) > threshold or abs(y - self.point_1[1]) > threshold:
                            # second click
                            self.point_2 = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag_bbox.anchor_being_dragged is not None:
                self.drag_bbox.handler_left_mouse_up(x, y)
    
    def _set_selected_bbox(self, set_class: bool):
        """Set the selected bounding box based on mouse position"""
        result = self.bbox_handler.set_selected_bbox(self.img_objects, self.mouse_x, self.mouse_y, self.drag_bbox)
        self.is_bbox_selected, self.selected_bbox = result
        if set_class and self.is_bbox_selected:
            # set class to the one of the selected bounding box
            cv2.setTrackbarPos(self.config.trackbar_class, self.config.window_name, self.img_objects[self.selected_bbox][0])
    
    def _save_bounding_box(self, annotation_paths: List[str], class_index: int, point_1: Tuple[int, int], point_2: Tuple[int, int], width: int, height: int):
        """Save a bounding box to annotation files"""
        for ann_path in annotation_paths:
            if '.txt' in ann_path:
                line = yolo_format(class_index, point_1, point_2, width, height)
                self._append_bb(ann_path, line, '.txt')
            elif '.xml' in ann_path:
                line = voc_format(self.class_list[class_index], point_1, point_2)
                self._append_bb(ann_path, line, '.xml')
    
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
            self.bbox_handler.write_xml(xml_str, ann_path)
    
    def _draw_close_icon(self, tmp_img: np.ndarray, x1_c: int, y1_c: int, x2_c: int, y2_c: int) -> np.ndarray:
        """Draw a close icon on the image"""
        red = (0, 0, 255)
        cv2.rectangle(tmp_img, (x1_c + 1, y1_c - 1), (x2_c, y2_c), red, -1)
        white = (255, 255, 255)
        cv2.line(tmp_img, (x1_c, y1_c), (x2_c, y2_c), white, 2)
        cv2.line(tmp_img, (x1_c, y2_c), (x2_c, y1_c), white, 2)
        return tmp_img
    
    def _draw_info_bb_selected(self, tmp_img: np.ndarray) -> np.ndarray:
        """Draw information for selected bounding boxes"""
        for idx, obj in enumerate(self.img_objects):
            ind, x1, y1, x2, y2 = obj
            if idx == self.selected_bbox:
                x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
                tmp_img = self._draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c)
        return tmp_img
    
    def run(self):
        """Run the main application loop"""
        # Initialize
        self._set_img_index(0)
        
        self._display_text('Welcome!\n Press [h] for help.', 4000)

        # Main loop
        while True:
            color = self.class_rgb[self.class_index].tolist()
            # Clone the img
            tmp_img = self.img.copy()
            height, width = tmp_img.shape[:2]
            if self.edges_on == True:
                # Draw edges
                tmp_img = draw_edges(tmp_img)
            # Draw vertical and horizontal guide lines
            tmp_img = draw_line(tmp_img, self.mouse_x, self.mouse_y, height, width, color)
            # Write selected class
            class_name = self.class_list[self.class_index]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            margin = 3
            text_width, text_height = cv2.getTextSize(class_name, font, font_scale, self.config.thickness)[0]
            tmp_img = cv2.rectangle(
                tmp_img, 
                (self.mouse_x + self.config.thickness, self.mouse_y - self.config.thickness), 
                (self.mouse_x + text_width + margin, self.mouse_y - text_height - margin), 
                complement_bgr(color), 
                -1
            )
            tmp_img = cv2.putText(
                tmp_img, 
                class_name, 
                (self.mouse_x + margin, self.mouse_y - margin), 
                font, 
                font_scale, 
                color, 
                self.config.thickness, 
                cv2.LINE_AA
            )
            # Get annotation paths
            img_path = self.image_path_list[self.img_index]
            annotation_paths = self.bbox_handler.get_annotation_paths(img_path, self.config.annotation_formats)
            
            if self.drag_bbox.anchor_being_dragged is not None:
                # Create a wrapper function for edit_bbox that includes all required parameters
                def edit_bbox_wrapper(obj_to_edit, action):
                    self.bbox_handler.edit_bbox(
                        obj_to_edit, action,
                        self.image_path_list, self.img_index,
                        self.config.annotation_formats,
                        self.img.shape[1], self.img.shape[0],
                        self.video_name_dict
                    )
                self.drag_bbox.handler_mouse_move(self.mouse_x, self.mouse_y, edit_bbox_wrapper)
            
            # Draw already done bounding boxes
            tmp_img = self.bbox_handler.draw_bboxes_from_file(
                tmp_img, annotation_paths, width, height, 
                self.img_objects, self.is_bbox_selected, self.selected_bbox, self.class_rgb
            )
            
            # If bounding box is selected add extra info
            if self.is_bbox_selected:
                tmp_img = self._draw_info_bb_selected(tmp_img)
            
            # If first click
            if self.point_1[0] != -1:
                # Draw partial bbox
                cv2.rectangle(tmp_img, self.point_1, (self.mouse_x, self.mouse_y), color, self.config.thickness)
                # If second click
                if self.point_2[0] != -1:
                    # Save the bounding box
                    self._save_bounding_box(
                        annotation_paths, self.class_index, 
                        self.point_1, self.point_2, width, height
                    )
                    # Reset the points
                    self.point_1 = (-1, -1)
                    self.point_2 = (-1, -1)

            cv2.imshow(self.config.window_name, tmp_img)
            pressed_key = cv2.waitKey(self.config.delay)

            if self.drag_bbox.anchor_being_dragged is None:
                ''' Key Listeners START '''
                if pressed_key == ord('a') or pressed_key == ord('d'):
                    # Show previous image key listener
                    if pressed_key == ord('a'):
                        self.img_index = decrease_index(self.img_index, self.last_img_index)
                    # Show next image key listener
                    elif pressed_key == ord('d'):
                        self.img_index = increase_index(self.img_index, self.last_img_index)
                    self._set_img_index(self.img_index)
                    cv2.setTrackbarPos(self.config.trackbar_img, self.config.window_name, self.img_index)
                elif pressed_key == ord('s') or pressed_key == ord('w'):
                    # Change down current class key listener
                    if pressed_key == ord('s'):
                        self.class_index = decrease_index(self.class_index, self.last_class_index)
                    # Change up current class key listener
                    elif pressed_key == ord('w'):
                        self.class_index = increase_index(self.class_index, self.last_class_index)
                    draw_line(tmp_img, self.mouse_x, self.mouse_y, height, width, color)
                    self._set_class_index(self.class_index)
                    cv2.setTrackbarPos(self.config.trackbar_class, self.config.window_name, self.class_index)
                    if self.is_bbox_selected:
                        obj_to_edit = self.img_objects[self.selected_bbox]
                        self.bbox_handler.edit_bbox(
                            obj_to_edit, 'change_class:{}'.format(self.class_index),
                            self.image_path_list, self.img_index,
                            self.config.annotation_formats,
                            self.img.shape[1], self.img.shape[0],
                            self.video_name_dict
                        )
                # Help key listener
                elif pressed_key == ord('h'):
                    text = ('[e] to show edges;\n'
                            '[q] to quit;\n'
                            '[a] or [d] to change Image;\n'
                            '[w] or [s] to change Class.\n'
                            )
                    self._display_text(text, 5000)
                # Show edges key listener
                elif pressed_key == ord('e'):
                    if self.edges_on == True:
                        self.edges_on = False
                        self._display_text('Edges turned OFF!', 1000)
                    else:
                        self.edges_on = True
                        self._display_text('Edges turned ON!', 1000)
                elif pressed_key == ord('p'):
                    # Check if the image is a frame from a video
                    is_from_video, video_name = self.bbox_handler.is_frame_from_video(img_path, self.video_name_dict)
                    if is_from_video:
                        # Get list of objects associated to that frame
                        object_list = self.img_objects[:]
                        # Remove the objects in that frame that are already in the `.json` file
                        json_file_path = '{}.json'.format(os.path.join(self.bbox_handler.tracker_dir, video_name))
                        file_exists, json_file_data = self.bbox_handler.get_json_file_data(json_file_path)
                        if file_exists:
                            object_list = self._remove_already_tracked_objects(object_list, img_path, json_file_data)
                        if len(object_list) > 0:
                            # Get list of frames following this image
                            next_frame_path_list = self.bbox_handler.get_next_frame_path_list(video_name, img_path, self.image_path_list, self.video_name_dict)
                            # Initial frame
                            init_frame = self.img.copy()
                            from .tracker import LabelTracker
                            label_tracker = LabelTracker(self.config.tracker_type, init_frame, next_frame_path_list, self.config)
                            for obj in object_list:
                                class_index = obj[0]
                                color = self.class_rgb[class_index].tolist()
                                label_tracker.start_tracker(json_file_data, json_file_path, img_path, obj, color, self.config.annotation_formats, self.bbox_handler, class_index, self.class_rgb)
                # Quit key listener
                elif pressed_key == ord('q'):
                    break
                ''' Key Listeners END '''

            if self.config.with_qt:
                # If window gets closed then quit
                if cv2.getWindowProperty(self.config.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

        cv2.destroyAllWindows()
    
    def _remove_already_tracked_objects(self, object_list: List, img_path: str, json_file_data: dict) -> List:
        """Remove objects that have already been tracked"""
        frame_data_dict = json_file_data['frame_data_dict']
        json_object_list = self.bbox_handler.get_json_file_object_list(img_path, frame_data_dict)
        # Copy the list since we will be deleting elements without restarting the loop
        temp_object_list = object_list[:]
        for obj in temp_object_list:
            obj_dict = self.bbox_handler.get_json_object_dict(obj, json_object_list)
            if obj_dict is not None:
                object_list.remove(obj)
                json_object_list.remove(obj_dict)
        return object_list





def main():
    """Main entry point for the application"""
    app = OpenLabelingApp()
    app.run()


if __name__ == "__main__":
    main()