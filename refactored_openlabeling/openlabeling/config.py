"""
Configuration module for OpenLabeling
"""
import argparse
import os
from typing import Dict, Any


class Config:
    """Configuration class for OpenLabeling application"""
    
    def __init__(self):
        self.args = self._parse_arguments()
        self.delay = 20  # keyboard delay (in milliseconds)
        self.with_qt = self._check_qt_support()
        
        # Directories and paths
        self.input_dir = self.args.input_dir
        self.output_dir = self.args.output_dir
        self.n_frames = self.args.n_frames
        self.tracker_type = self.args.tracker
        self.thickness = self.args.thickness
        
        # Drawing settings
        self.draw_from_pascal = self.args.draw_from_PASCAL_files
        self.window_name = 'OpenLabeling'
        self.annotation_formats = {'PASCAL_VOC': '.xml', 'YOLO_darknet': '.txt'}
        
        # UI elements
        self.trackbar_img = 'Image'
        self.trackbar_class = 'Class'
        
        # Resizing anchors
        self.line_thickness = self.args.thickness
        self.resizing_anchor_size = self.line_thickness * 2

    def _parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(description='Open-source image labeling tool')
        parser.add_argument(
            '-i', '--input_dir', 
            default='input', 
            type=str, 
            help='Path to input directory'
        )
        parser.add_argument(
            '-o', '--output_dir', 
            default='output', 
            type=str, 
            help='Path to output directory'
        )
        parser.add_argument(
            '-t', '--thickness', 
            default=1, 
            type=int, 
            help='Bounding box and cross line thickness'
        )
        parser.add_argument(
            '--draw-from-PASCAL-files', 
            action='store_true', 
            help='Draw bounding boxes from the PASCAL files'
        )
        parser.add_argument(
            '--tracker', 
            default='KCF', 
            type=str, 
            help="tracker_type being used: ['CSRT', 'KCF','MOSSE', 'MIL', 'BOOSTING', 'MEDIANFLOW', 'TLD', 'GOTURN', 'DASIAMRPN']"
        )
        parser.add_argument(
            '-n', '--n_frames', 
            default=200, 
            type=int, 
            help='number of frames to track object for'
        )
        return parser.parse_args()

    def _check_qt_support(self) -> bool:
        """Check if OpenCV supports Qt"""
        import cv2
        with_qt = False
        try:
            cv2.namedWindow('Test')
            cv2.displayOverlay('Test', 'Test QT', 500)
            with_qt = True
        except cv2.error:
            print('-> Please ignore this error message\n')
        finally:
            cv2.destroyAllWindows()
        return with_qt