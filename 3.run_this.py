import glob
import os

import numpy as np
import cv2

class_index = 0
img_index = 0
img = None
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
    x_center = (point_1[0] + point_2[0]) / float(2.0 * height)
    y_center = (point_1[1] + point_2[1]) / float(2.0 * width)
    x_width = float(abs(point_2[0] - point_1[0])) / height
    y_height = float(abs(point_2[1] - point_1[1])) / width
    return str(class_index) + " " + str(x_center) \
       + " " + str(y_center) + " " + str(x_width) + " " + str(y_height)


def get_txt_path(img_path):
    img_type = img_path.split('.')[-1]
    return img_path.replace(img_type, 'txt')


def save_bb(text_path, line):
    with open(text_path, 'a') as myfile:
        myfile.write(line + "\n") # append line


def yolo_to_x_y(x_center, y_center, x_width, y_height, width, height):
    x_center *= width
    y_center *= height
    x_width *= width
    y_height *= height
    x_width /= 2.0
    y_height /= 2.0
    return int(x_center - x_width), int(y_center - y_height), int(x_center + x_width), int(y_center + y_height)

def draw_bboxes_from_file(tmp_img, txt_path, width, height):
    if os.path.isfile(txt_path):
        with open(txt_path) as f:
            content = f.readlines()
        for line in content:
            values_str = line.split()
            class_index, x_center, y_center, x_width, y_height = map(float, values_str)
            # convert yolo to points
            x1, y1, x2, y2 = yolo_to_x_y(x_center, y_center, x_width, y_height, width, height)
            color = class_rgb[int(class_index)]
            cv2.rectangle(tmp_img, (x1, y1), (x2, y2), color, 2)
    return tmp_img

# mouse callback function
def draw_roi(event, x, y, flags, param):
    global mouse_x, mouse_y, point_1, point_2
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y
    elif event == cv2.EVENT_LBUTTONDOWN:
        if point_1[0] is -1:
            # first click
            cv2.displayOverlay(WINDOW_NAME, "Currently selected label:\n"
                                    "" + class_list[class_index] + "", 2000)
            point_1 = (x, y)
        else:
            # second click
            point_2 = (x, y)

# load img list
image_list = glob.glob('1.insert_images_here/*.jpg')
image_list.extend(glob.glob('1.insert_images_here/*.jpeg'))
print(image_list)
last_img_index = len(image_list) - 1

# load class list
with open('2.insert_class_list_here.txt') as f:
    class_list = f.read().splitlines()
#print(class_list)
last_class_index = len(class_list) - 1

# random rgb for each class
class_rgb = np.random.random_integers(low=0, high=255, size=(3,len(class_list)))

# create window
WINDOW_NAME = 'openbbox'
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

cv2.displayOverlay(WINDOW_NAME, "Welcome to openbbox!\n Press [h] for help.", 4000)
print(" Welcome to openbbox!\n Select the window and press [h] for help.")

# loop
while True:
    # clone the img
    tmp_img = img.copy()
    width, height = tmp_img.shape[:2]
    if edges_on == True:
        # draw edges
        tmp_img = draw_edges(tmp_img)
    # draw vertical yellow lines
    draw_line(tmp_img, mouse_x, mouse_y, width, height) # show vertical and horizontal line
    img_path = image_list[img_index]
    txt_path = get_txt_path(img_path)
    # draw already done bounding boxes
    tmp_img = draw_bboxes_from_file(tmp_img, txt_path, width, height)
    # if first click
    if point_1[0] is not -1:
        color = class_rgb[class_index]
        # draw partial bbox
        cv2.rectangle(tmp_img, point_1, (mouse_x, mouse_y), color, 2)
        # if second click
        if point_2[0] is not -1:
            # save the bounding box
            line = yolo_format(class_index, point_1, point_2, width, height)
            save_bb(txt_path, line)
            # reset the points
            point_1 = (-1, -1)
            point_2 = (-1, -1)

    cv2.imshow(WINDOW_NAME, tmp_img)
    pressed_key = cv2.waitKey(50)

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
