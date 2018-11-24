import argparse
import textwrap
import glob
import os

import numpy as np
import cv2


DELAY = 20 # keyboard delay (in milliseconds)
WITH_QT = True
try:
    cv2.namedWindow("Test")
    cv2.displayOverlay("Test", "Test QT", 1000)
except:
    WITH_QT = False
cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description='YOLO v2 Bounding Box Tool')
parser.add_argument('--format', default='yolo', type=str, choices=['yolo', 'voc'], help="Bounding box format")
parser.add_argument('--sort', action='store_true', help="If true, shows images in order.")
parser.add_argument('--cross-thickness', default='1', type=int, help="Cross thickness")
parser.add_argument('--bbox-thickness', default='1', type=int, help="Bounding box thickness")
parser.add_argument('--img_dir', default='images/', type=str, help="Path to Image Directory")
parser.add_argument('--bb_dir', default='bbox_txt/', type=str, help="Path to Bbox Directory")
args = parser.parse_args()

class_index = 0
img_index = 0
img = None
img_objects = []
bb_dir = args.bb_dir

WINDOW_NAME = 'Bounding Box Labeler'
TRACKBAR_IMG = 'Image'
TRACKBAR_CLASS = 'Class'

# selected bounding box
prev_was_double_click = False
is_bbox_selected = False
selected_bbox = -1

mouse_x = 0
mouse_y = 0
point_1 = (-1, -1)
point_2 = (-1, -1)


def set_img_index(x):
    global img_index, img
    img_index = x
    img_path = image_list[img_index]
    img = cv2.imread(img_path)
    if WITH_QT:
        cv2.displayOverlay(WINDOW_NAME, "Showing image "
                                    "" + str(img_index) + "/"
                                    "" + str(last_img_index), 1000)
    else:
        print("Showing image "
                "" + str(img_index) + "/"
                "" + str(last_img_index) + " path:" + img_path)


def set_class_index(x):
    global class_index
    class_index = x
    if WITH_QT:
        cv2.displayOverlay(WINDOW_NAME, "Selected class "
                                "" + str(class_index) + "/"
                                "" + str(last_class_index) + ""
                                "\n " + class_list[class_index], 3000)
    else:
        print("Selected class :" + class_list[class_index])


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
    cv2.line(img, (x, 0), (x, height), color, thickness=args.cross_thickness)
    cv2.line(img, (0, y), (width, y), color, thickness=args.cross_thickness)


def yolo_format(class_index, point_1, point_2, width, height):
    # YOLO wants everything normalized
    # Order: class x_center y_center x_width y_height
    x_center = (point_1[0] + point_2[0]) / float(2.0 * width)
    y_center = (point_1[1] + point_2[1]) / float(2.0 * height)
    x_width = float(abs(point_2[0] - point_1[0])) / width
    y_height = float(abs(point_2[1] - point_1[1])) / height
    items = map(str, [class_index, x_center, y_center, x_width, y_height])
    return ' '.join(items)


def voc_format(class_index, point_1, point_2):
    # Order: xmin ymin xmax ymax class
    # Top left pixel is (1, 1) in VOC
    xmin, ymin = min(point_1[0], point_2[0]) + 1, min(point_1[1], point_2[1]) + 1
    xmax, ymax = max(point_1[0], point_2[0]) + 1, max(point_1[1], point_2[1]) + 1
    items = map(str, [class_index, xmin, ymin, xmax, ymax])
    return ' '.join(items)


def get_txt_path(img_path):
    img_name = os.path.basename(os.path.normpath(img_path))
    img_type = img_path.split('.')[-1]
    return bb_dir + img_name.replace(img_type, 'txt')


def save_bb(txt_path, line):
    with open(txt_path, 'a') as myfile:
        myfile.write(line + "\n") # append line


def yolo_to_x_y(x_center, y_center, x_width, y_height, width, height):
    x_center *= width
    y_center *= height
    x_width *= width
    y_height *= height
    x_width /= 2.0
    y_height /= 2.0
    return int(x_center - x_width), int(y_center - y_height), int(x_center + x_width), int(y_center + y_height)


def draw_text(tmp_img, text, center, color, size):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(tmp_img, text, center, font, 0.6, color, size, cv2.LINE_AA)
    return tmp_img


def draw_bboxes_from_file(tmp_img, txt_path, width, height):
    global img_objects
    img_objects = []
    if os.path.isfile(txt_path):
        with open(txt_path) as f:
            content = f.readlines()
        for line in content:
            values_str = line.split()
            if args.format == 'yolo':
                class_index, x_center, y_center, x_width, y_height = map(float, values_str)
                class_index = int(class_index)
                # convert yolo to points
                x1, y1, x2, y2 = yolo_to_x_y(x_center, y_center, x_width, y_height, width, height)
                if x_center == int(x_center):
                    error = ("You selected the 'yolo' format but your labels "
                             "seem to be in a different format. Consider "
                             "removing your old label files.")
                    raise Exception(textwrap.fill(error, 70))
            elif args.format == 'voc':
                try:
                    class_index, x1, y1, x2, y2 = map(int, values_str)
                except ValueError:
                    error = ("You selected the 'voc' format but your labels "
                             "seem to be in a different format. Consider "
                             "removing your old label files.")
                    raise Exception(textwrap.fill(error, 70))
                x1, y1, x2, y2 = x1-1, y1-1, x2-1, y2-1
            img_objects.append([class_index, x1, y1, x2, y2])
            color = class_rgb[class_index].tolist()
            cv2.rectangle(tmp_img, (x1, y1), (x2, y2), color, thickness=args.bbox_thickness)
            tmp_img = draw_text(tmp_img, class_list[class_index], (x1, y1 - 5), color, args.bbox_thickness)
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
        ind, x1, y1, x2, y2 = obj
        if is_mouse_inside_points(x1, y1, x2, y2):
            is_bbox_selected = True
            tmp_area = get_bbox_area(x1, y1, x2, y2)
            if tmp_area < smallest_area or smallest_area == -1:
                smallest_area = tmp_area
                selected_bbox = idx
                if set_class:
                    # set class to the one of the selected bounding box
                    cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, ind)


def mouse_inside_delete_button():
    for idx, obj in enumerate(img_objects):
        if idx == selected_bbox:
            _ind, x1, y1, x2, y2 = obj
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            if is_mouse_inside_points(x1_c, y1_c, x2_c, y2_c):
                return True
    return False


def edit_selected_bbox(delete, new_class):
    img_path = image_list[img_index]
    txt_path = get_txt_path(img_path)

    with open(txt_path, "r") as old_file:
        lines = old_file.readlines()

    with open(txt_path, "w") as new_file:
        counter = 0
        for line in lines:
            if counter != selected_bbox:
                # copy the other bounding boxes
                new_file.write(line)
            elif delete != True:
                items = line.split()
                items[0] = str(class_index)
                new_file.write(" ".join(items) + '\n')
            counter += 1


# mouse callback function
def mouse_listener(event, x, y, flags, param):
    global is_bbox_selected, prev_was_double_click, mouse_x, mouse_y, point_1, point_2

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        prev_was_double_click = True
        #print("Double click")
        point_1 = (-1, -1)
        # if clicked inside a bounding box we set that bbox
        set_class = True
        set_selected_bbox(set_class)
    # AlexeyGy change: delete via right-click
    elif event == cv2.EVENT_RBUTTONDOWN:
        set_class = False
        set_selected_bbox(set_class)
        if is_bbox_selected:
            edit_selected_bbox(True, -1)
            is_bbox_selected = False
    elif event == cv2.EVENT_LBUTTONDOWN:
        if prev_was_double_click:
            #print("Finish double click")
            prev_was_double_click = False

        #print("Normal left click")
        is_mouse_inside_delete_button = mouse_inside_delete_button()
        if point_1[0] is -1:
            if is_bbox_selected:
                if is_mouse_inside_delete_button:
                    # the user wants to delete the bbox
                    #print("Delete bbox")
                    edit_selected_bbox(True, -1)
                is_bbox_selected = False
            else:
                # first click (start drawing a bounding box or delete an item)
                point_1 = (x, y)
        else:
            # minimal size for bounding box to avoid errors
            threshold = 20
            if abs(x - point_1[0]) > threshold or abs(y - point_1[1]) > threshold:
                # second click
                point_2 = (x, y)


def is_mouse_inside_points(x1, y1, x2, y2):
    return mouse_x > x1 and mouse_x < x2 and mouse_y > y1 and mouse_y < y2


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


def draw_info_bb_selected(tmp_img):
    for idx, obj in enumerate(img_objects):
        ind, x1, y1, x2, y2 = obj
        if idx == selected_bbox:
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c)
    return tmp_img


# load all images (with multiple extensions) from a directory using OpenCV
img_dir = args.img_dir
image_list = []
for f in os.listdir(img_dir):
    f_path = os.path.join(img_dir, f)
    test_img = cv2.imread(f_path)
    if test_img is not None:
        image_list.append(f_path)

#print(image_list)
image_list.sort()
if not args.sort:
    np.random.seed(123)  # Keep random img order consistent
    np.random.shuffle(image_list)
last_img_index = len(image_list) - 1

if not os.path.exists(bb_dir):
    os.makedirs(bb_dir)

# create empty .txt file for each of the images if it doesn't exist already
for img_path in image_list:
    txt_path = get_txt_path(img_path)
    if not os.path.isfile(txt_path):
         open(txt_path, 'a').close()

# load class list
with open('class_list.txt') as f:
    class_list = f.read().splitlines()
#print(class_list)
last_class_index = len(class_list) - 1

# Make the class colors the same each session
# The colors are in BGR order because we're using OpenCV
class_rgb = [
    (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0),
    (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128)]
class_rgb = np.array(class_rgb)
# If there are still more classes, add new colors randomly
num_colors_missing = len(class_list) - len(class_rgb)
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

if WITH_QT:
    cv2.displayOverlay(WINDOW_NAME, "Welcome!\n Press [h] for help.", 4000)
print(" Welcome!\n Select the window and press [h] for help.")

# loop
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
    img_path = image_list[img_index]
    txt_path = get_txt_path(img_path)
    # draw already done bounding boxes
    tmp_img = draw_bboxes_from_file(tmp_img, txt_path, width, height)
    # if bounding box is selected add extra info
    if is_bbox_selected:
        tmp_img = draw_info_bb_selected(tmp_img)
    # if first click
    if point_1[0] is not -1:
        # draw partial bbox
        cv2.rectangle(tmp_img, point_1, (mouse_x, mouse_y), color, thickness=args.bbox_thickness)
        # if second click
        if point_2[0] is not -1:
            # save the bounding box
            if args.format == 'yolo':
                line = yolo_format(class_index, point_1, point_2, width, height)
            elif args.format == 'voc':
                line = voc_format(class_index, point_1, point_2)
            save_bb(txt_path, line)
            # reset the points
            point_1 = (-1, -1)
            point_2 = (-1, -1)
        else:
            if WITH_QT:
                cv2.displayOverlay(WINDOW_NAME, "Selected label: " + class_list[class_index] + ""
                                    "\nPress [w] or [s] to change.", 120)

    cv2.imshow(WINDOW_NAME, tmp_img)
    pressed_key = cv2.waitKey(DELAY)

    """ Key Listeners START """
    if pressed_key == ord('a') or pressed_key == ord('d'):
        # show previous image key listener
        if pressed_key == ord('a'):
            img_index = decrease_index(img_index, last_img_index)
        # show next image key listener
        elif pressed_key == ord('d'):
            img_index = increase_index(img_index, last_img_index)
        cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)
    elif pressed_key == ord('s') or pressed_key == ord('w'):
        # change down current class key listener
        if pressed_key == ord('s'):
            class_index = decrease_index(class_index, last_class_index)
        # change up current class key listener
        elif pressed_key == ord('w'):
            class_index = increase_index(class_index, last_class_index)
        draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
        cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_index)
        if is_bbox_selected:
            edit_selected_bbox(False, class_index)
    # help key listener
    elif pressed_key == ord('h'):
        if WITH_QT:
            cv2.displayOverlay(WINDOW_NAME, "[e] to show edges;\n"
                                "[q] to quit;\n"
                                "[a] or [d] to change Image;\n"
                                "[w] or [s] to change Class.\n"
                                "%s" % img_path, 6000)
        else:
            print("[e] to show edges;\n"
                    "[q] to quit;\n"
                    "[a] or [d] to change Image;\n"
                    "[w] or [s] to change Class.\n"
                    "%s" % img_path)
    # show edges key listener
    elif pressed_key == ord('e'):
        if edges_on == True:
            edges_on = False
            if WITH_QT:
                cv2.displayOverlay(WINDOW_NAME, "Edges turned OFF!", 1000)
            else:
                print("Edges turned OFF!")
        else:
            edges_on = True
            if WITH_QT:
                cv2.displayOverlay(WINDOW_NAME, "Edges turned ON!", 1000)
            else:
                print("Edges turned ON!")
    # quit key listener
    elif pressed_key == ord('q'):
        break
    """ Key Listeners END """

    if WITH_QT:
        # if window gets closed then quit
        if cv2.getWindowProperty(WINDOW_NAME,cv2.WND_PROP_VISIBLE) < 1:
            break

cv2.destroyAllWindows()
