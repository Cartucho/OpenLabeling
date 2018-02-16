import argparse
import glob
import os

import numpy as np
import cv2


parser = argparse.ArgumentParser(description='YOLO v2 Bounding Box Tool')
parser.add_argument('format', default='yolo', type=str, help="Bounding box format. Default YOLO. Options: ['yolo', 'voc']")
args = parser.parse_args()

class_index = 0
img_index = 0
img = None
img_objects = []
bb_dir = "bbox_txt/"

mouse_x = 0
mouse_y = 0
point_1 = (-1, -1)
point_2 = (-1, -1)


def change_img_index(x):
    global img_index, img
    img_index = x
    img_path = image_list[img_index]
    img = cv2.imread(img_path)
    cv2.displayOverlay(WINDOW_NAME, "Showing image "
                                    "" + str(img_index) + "/"
                                    "" + str(last_img_index), 1000)


def change_class_index(x):
    global class_index
    class_index = x
    cv2.displayOverlay(WINDOW_NAME, "Selected class "
                                "" + str(class_index) + "/"
                                "" + str(last_class_index) + ""
                                "\n " + class_list[class_index],3000)


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


def draw_line(img, x, y, height, width):
    cv2.line(img, (x, 0), (x, height), (0, 255, 255))
    cv2.line(img, (0, y), (width, y), (0, 255, 255))


def yolo_format(class_index, point_1, point_2, height, width):
    # YOLO wants everything normalized
    # Order: class x_center y_center x_width y_height
    x_center = (point_1[0] + point_2[0]) / float(2.0 * height)
    y_center = (point_1[1] + point_2[1]) / float(2.0 * width)
    x_width = float(abs(point_2[0] - point_1[0])) / height
    y_height = float(abs(point_2[1] - point_1[1])) / width
    return str(class_index) + " " + str(x_center) \
        + " " + str(y_center) + " " + str(x_width) + " " + str(y_height)


def voc_format(class_index, point_1, point_2):
    # Order: xmin ymin xmax ymax class
    items = list(point_1) + list(point_2) + [class_index]
    items = map(str, items)
    return ' '.join(items)


def get_txt_path(img_path):
    img_name = img_path.split('/')[-1]
    img_type = img_path.split('.')[-1]
    return bb_dir + img_name.replace(img_type, 'txt')


def save_bb(txt_path, line):
    with open(txt_path, 'a') as myfile:
        myfile.write(line + "\n") # append line


def delete_bb(txt_path, line_index):
    with open(txt_path, "r") as old_file:
        lines = old_file.readlines()

    with open(txt_path, "w") as new_file:
        counter = 0
        for line in lines:
            if counter is not line_index:
                new_file.write(line)
            counter += 1


def yolo_to_x_y(x_center, y_center, x_width, y_height, width, height):
    x_center *= width
    y_center *= height
    x_width *= width
    y_height *= height
    x_width /= 2.0
    y_height /= 2.0
    return int(x_center - x_width), int(y_center - y_height), int(x_center + x_width), int(y_center + y_height)


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
            elif args.format == 'voc':
                x1, y1, x2, y2, class_index = map(int, values_str)
            else:
                raise Exception("Unknown bounding box format.")
            img_objects.append([class_index, x1, y1, x2, y2])
            color = class_rgb[class_index].tolist()
            cv2.rectangle(tmp_img, (x1, y1), (x2, y2), color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(tmp_img, class_list[class_index], (x1, y1 - 5), font, 0.6, color, 2, cv2.LINE_AA)
    return tmp_img


# mouse callback function
def draw_roi(event, x, y, flags, param):
    global mouse_x, mouse_y, point_1, point_2
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y
    elif event == cv2.EVENT_LBUTTONDOWN:
        if point_1[0] is -1:
            # first click (start drawing a bounding box or delete an item)
            point_1 = (x, y)
        else:
            # second click
            point_2 = (x, y)


def is_mouse_inside_points(x1, y1, x2, y2):
    return mouse_x > x1 and mouse_x < x2 and mouse_y > y1 and mouse_y < y2


def get_close_icon(x1, y1, x2, y2):
    percentage = 0.1
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


def draw_info_if_bb_selected(tmp_img):
    for obj in img_objects:
        ind, x1, y1, x2, y2 = obj
        if is_mouse_inside_points(x1, y1, x2, y2):
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            tmp_img = draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c)
    return tmp_img


# load img list
img_dir = "images/"
image_list = glob.glob(img_dir +'*.jpg')
image_list.extend(glob.glob(img_dir + '*.jpeg'))
#print(image_list)
last_img_index = len(image_list) - 1

# load class list
with open('class_list.txt') as f:
    class_list = f.read().splitlines()
#print(class_list)
last_class_index = len(class_list) - 1

# random rgb for each class
class_rgb = np.random.random_integers(low=0, high=255, size=(len(class_list),3))

# create window
WINDOW_NAME = 'Bounding Box Labeler'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_NAME, 1000, 700)
cv2.setMouseCallback(WINDOW_NAME, draw_roi)

# selected image
TRACKBAR_IMG = 'Image'
cv2.createTrackbar(TRACKBAR_IMG, WINDOW_NAME, 0, last_img_index, change_img_index)

# selected class
TRACKBAR_CLASS = 'Class'
cv2.createTrackbar(TRACKBAR_CLASS, WINDOW_NAME, 0, last_class_index, change_class_index)

# initialize
change_img_index(0)
edges_on = False

cv2.displayOverlay(WINDOW_NAME, "Welcome!\n Press [h] for help.", 4000)
print(" Welcome!\n Select the window and press [h] for help.")

if not os.path.exists(bb_dir):
    os.makedirs(bb_dir)

# loop
while True:
    # clone the img
    tmp_img = img.copy()
    height, width = tmp_img.shape[:2]
    if edges_on == True:
        # draw edges
        tmp_img = draw_edges(tmp_img)
    # draw vertical and horizong yellow guide lines
    draw_line(tmp_img, mouse_x, mouse_y, height, width)
    img_path = image_list[img_index]
    txt_path = get_txt_path(img_path)
    # draw already done bounding boxes
    tmp_img = draw_bboxes_from_file(tmp_img, txt_path, width, height)
    # if bounding box is selected add extra info
    tmp_img = draw_info_if_bb_selected(tmp_img)
    # if first click
    if point_1[0] is not -1:
        removed_an_object = False
        # if clicked inside a delete button, then remove that object
        for obj in img_objects:
            ind, x1, y1, x2, y2 = obj
            x1, y1, x2, y2 = get_close_icon(x1, y1, x2, y2)
            if is_mouse_inside_points(x1, y1, x2, y2):
                # remove that object
                delete_bb(txt_path, img_objects.index(obj))
                removed_an_object = True
                point_1 = (-1, -1)
                break

        if not removed_an_object:
            color = class_rgb[class_index].tolist()
            # draw partial bbox
            cv2.rectangle(tmp_img, point_1, (mouse_x, mouse_y), color, 2)
            # if second click
            if point_2[0] is not -1:
                # save the bounding box
                if args.format == 'yolo':
                    line = yolo_format(class_index, point_1, point_2, width, height)
                elif args.format == 'voc':
                    line = voc_format(class_index, point_1, point_2)
                else:
                    raise Exception("Unknown bounding box format.")
                save_bb(txt_path, line)
                # reset the points
                point_1 = (-1, -1)
                point_2 = (-1, -1)
            else:
                cv2.displayOverlay(WINDOW_NAME, "Selected label: " + class_list[class_index] + ""
                                        "\nPress [w] or [s] to change.", 120)

    cv2.imshow(WINDOW_NAME, tmp_img)
    pressed_key = cv2.waitKey(100)

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
        cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_index)

    # help key listener
    elif pressed_key == ord('h'):
        cv2.displayOverlay(WINDOW_NAME, "[e] to show edges;\n"
                                "[q] to quit;\n"
                                "[a] or [d] to change Image;\n"
                                "[w] or [s] to change Class.", 6000)
    # show edges key listener
    elif pressed_key == ord('e'):
        if edges_on == True:
            edges_on = False
            cv2.displayOverlay(WINDOW_NAME, "Edges turned OFF!", 1000)
        else:
            cv2.displayOverlay(WINDOW_NAME, "Edges turned ON!", 1000)
            edges_on = True
    # quit key listener
    elif pressed_key == ord('q'):
        break
    """ Key Listeners END """

    # if window gets closed then quit
    if cv2.getWindowProperty(WINDOW_NAME,cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
