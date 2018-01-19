import glob

import numpy as np
import cv2

class_index = 0
img_index = 0
img = None

def change_img_index(x):
    global img_index, img
    img_index = x
    img_path = image_list[img_index]
    img = cv2.imread(img_path)


def change_class_index(x):
    global class_index
    class_index = x


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

# load img list
image_list = glob.glob('1.insert_images_here/*')
print(image_list)

# load class list
with open('2.insert_class_list_here.txt') as f:
    class_list = f.read().splitlines()
print(class_list)

# random rgb for each class
class_rgb = np.random.random_integers(low=0, high=255, size=(3,len(class_list)))

# create window
WINDOW_NAME = 'fastbbox'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_NAME, 1000, 600)

# selected image
TRACKBAR_IMG = 'Image'
cv2.createTrackbar(TRACKBAR_IMG, WINDOW_NAME, 0, len(image_list) - 1, change_img_index)

# selected class
TRACKBAR_CLASS = 'Class'
cv2.createTrackbar(TRACKBAR_CLASS, WINDOW_NAME, 0, len(class_list) - 1, change_class_index)

# initialize
change_img_index(0)
edges_on = False

# loop
while True:
    # clone the img
    tmp_img = img.copy()
    if edges_on == True:
        # draw edges
        tmp_img = draw_edges(tmp_img)

    cv2.imshow(WINDOW_NAME, tmp_img)
    pressed_key = cv2.waitKey(50)

    # show previous image key listener
    if pressed_key == ord('a'):
        img_index = decrease_index(img_index, len(image_list) - 1)
        cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)
    # show next image key listener
    elif pressed_key == ord('d'):
        img_index = increase_index(img_index, len(image_list) - 1)
        cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)
    # change down current class key listener
    elif pressed_key == ord('s'):
        class_index = decrease_index(class_index, len(class_list) - 1)
        cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_index)
    # change up current class key listener
    elif pressed_key == ord('w'):
        class_index = increase_index(class_index, len(class_list) - 1)
        cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_index)
    # show edges key listener
    elif pressed_key == ord('e'):
        if edges_on == True:
            edges_on = False
        else:
            edges_on = True
    # q - quit key listener
    elif pressed_key == ord('q'):
        break

    # if closed window then close code
    if cv2.getWindowProperty(WINDOW_NAME,cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
