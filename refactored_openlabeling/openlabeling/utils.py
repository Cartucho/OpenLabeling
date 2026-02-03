"""
Utility functions for OpenLabeling
"""
import os
import re
import json
from typing import Tuple, List, Optional, Union
from pathlib import Path
import cv2
import numpy as np


def point_in_rect(p_x: float, p_y: float, r_x_left: float, r_y_top: float, r_x_right: float, r_y_bottom: float) -> bool:
    """
    Check if a point belongs to a rectangle
    """
    return r_x_left <= p_x <= r_x_right and r_y_top <= p_y <= r_y_bottom


def get_bbox_area(x1: int, y1: int, x2: int, y2: int) -> int:
    """Calculate the area of a bounding box"""
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width * height


def natural_sort_key(s: str, _nsre: re.Pattern = re.compile('([0-9]+)')) -> List[Union[int, str]]:
    """
    Natural sort key function to handle numeric values in strings properly
    """
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]


def nonblank_lines(f) -> str:
    """
    Generator that yields non-blank lines from a file
    """
    for line in f:
        stripped_line = line.rstrip()
        if stripped_line:
            yield stripped_line


def yolo_format(class_index: int, point_1: Tuple[int, int], point_2: Tuple[int, int], width: int, height: int) -> str:
    """
    Convert bounding box coordinates to YOLO format
    Order: class x_center y_center x_width y_height
    """
    x_center = float((point_1[0] + point_2[0]) / (2.0 * width))
    y_center = float((point_1[1] + point_2[1]) / (2.0 * height))
    x_width = float(abs(point_2[0] - point_1[0])) / width
    y_height = float(abs(point_2[1] - point_1[1])) / height
    items = map(str, [class_index, x_center, y_center, x_width, y_height])
    return ' '.join(items)


def voc_format(class_name: str, point_1: Tuple[int, int], point_2: Tuple[int, int]) -> Tuple[str, str, str, str, str]:
    """
    Convert bounding box coordinates to VOC format
    Order: class_name xmin ymin xmax ymax
    """
    xmin, ymin = min(point_1[0], point_2[0]), min(point_1[1], point_2[1])
    xmax, ymax = max(point_1[0], point_2[0]), max(point_1[1], point_2[1])
    items = map(str, [class_name, xmin, ymin, xmax, ymax])
    return tuple(items)


def yolo_to_voc(x_center: float, y_center: float, x_width: float, y_height: float, width: int, height: int) -> Tuple[int, int, int, int]:
    """
    Convert YOLO format to VOC format
    """
    x_center *= float(width)
    y_center *= float(height)
    x_width *= float(width)
    y_height *= float(height)
    x_width /= 2.0
    y_height /= 2.0
    xmin = int(round(x_center - x_width))
    ymin = int(round(y_center - y_height))
    xmax = int(round(x_center + x_width))
    ymax = int(round(y_center + y_height))
    return xmin, ymin, xmax, ymax


def complement_bgr(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Get complementary BGR color
    """
    lo = min(color)
    hi = max(color)
    k = lo + hi
    return tuple(k - u for u in color)


def get_close_icon(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
    """
    Calculate close icon coordinates for a bounding box
    """
    percentage = 0.05
    height = -1
    while height < 15 and percentage < 1.0:
        height = int((y2 - y1) * percentage)
        percentage += 0.1
    return (x2 - height), y1, x2, (y1 + height)


def get_anchors_rectangles(xmin: int, ymin: int, xmax: int, ymax: int, anchor_size: int) -> dict:
    """
    Calculate anchor rectangles for a bounding box
    """
    anchor_list = {}

    mid_x = (xmin + xmax) / 2
    mid_y = (ymin + ymax) / 2

    L_ = [xmin - anchor_size, xmin + anchor_size]
    M_ = [mid_x - anchor_size, mid_x + anchor_size]
    R_ = [xmax - anchor_size, xmax + anchor_size]
    _T = [ymin - anchor_size, ymin + anchor_size]
    _M = [mid_y - anchor_size, mid_y + anchor_size]
    _B = [ymax - anchor_size, ymax + anchor_size]

    anchor_list['LT'] = [L_[0], _T[0], L_[1], _T[1]]
    anchor_list['MT'] = [M_[0], _T[0], M_[1], _T[1]]
    anchor_list['RT'] = [R_[0], _T[0], R_[1], _T[1]]
    anchor_list['LM'] = [L_[0], _M[0], L_[1], _M[1]]
    anchor_list['RM'] = [R_[0], _M[0], R_[1], _M[1]]
    anchor_list['LB'] = [L_[0], _B[0], L_[1], _B[1]]
    anchor_list['MB'] = [M_[0], _B[0], M_[1], _B[1]]
    anchor_list['RB'] = [R_[0], _B[0], R_[1], _B[1]]

    return anchor_list


def draw_edges(img: np.ndarray) -> np.ndarray:
    """
    Draw edges on an image using bilateral filter and Canny edge detection
    """
    blur = cv2.bilateralFilter(img, 3, 75, 75)
    edges = cv2.Canny(blur, 150, 250, 3)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Overlap image and edges together
    img = np.bitwise_or(img, edges)
    return img


def draw_line(img: np.ndarray, x: int, y: int, height: int, width: int, color: Tuple[int, int, int]) -> np.ndarray:
    """
    Draw crosshair lines on an image
    """
    cv2.line(img, (x, 0), (x, height), color, 1)
    cv2.line(img, (0, y), (width, y), color, 1)
    return img


def increase_index(current_index: int, last_index: int) -> int:
    """
    Increase index with wrap-around
    """
    current_index += 1
    if current_index > last_index:
        current_index = 0
    return current_index


def decrease_index(current_index: int, last_index: int) -> int:
    """
    Decrease index with wrap-around
    """
    current_index -= 1
    if current_index < 0:
        current_index = last_index
    return current_index