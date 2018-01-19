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

# load img list
image_list = glob.glob('1.insert_images_here/*')
#print(image_list)
last_img_index = len(image_list) - 1

# load class list
with open('2.insert_class_list_here.txt') as f:
    class_list = f.read().splitlines()
#print(class_list)
last_class_index = len(class_list) - 1

# random rgb for each class
class_rgb = np.random.random_integers(low=0, high=255, size=(3,len(class_list)))

# create window
WINDOW_NAME = 'fastbbox'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_NAME, 1000, 600)

# selected image
TRACKBAR_IMG = 'Image'
cv2.createTrackbar(TRACKBAR_IMG, WINDOW_NAME, 0, last_img_index, change_img_index)

# selected class
TRACKBAR_CLASS = 'Class'
cv2.createTrackbar(TRACKBAR_CLASS, WINDOW_NAME, 0, last_class_index, change_class_index)

# initialize
change_img_index(0)
edges_on = False

cv2.displayOverlay(WINDOW_NAME, "Welcome to fastbbox!\n Press [h] for help.", 4000)
print(" Welcome to fastbbox!\n Select the window and press [h] for help.")

# loop
while True:
    # clone the img
    tmp_img = img.copy()
    if edges_on == True:
        # draw edges
        tmp_img = draw_edges(tmp_img)

    cv2.imshow(WINDOW_NAME, tmp_img)
    pressed_key = cv2.waitKey(50)

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

    # if closed window then close code
    if cv2.getWindowProperty(WINDOW_NAME,cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
