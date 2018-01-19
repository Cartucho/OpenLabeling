import glob

import numpy as np
import cv2

def shrink_img(img):
    
    

# load class list
with open('input/class_list.txt') as f:
    CLASSES = f.read().splitlines()
print(CLASSES)

# one color for each class
CLASS_COLOR = np.random.random_integers(low=0, high=255, size=(3,len(CLASSES))) # random rgb for each class

# load img list
IMAGES = glob.glob('input/images/*')
print(IMAGES)

WINDOW_NAME = 'fastbbox'

for img_path in IMAGES:
    img = cv2.imread(img_path)
    img = shrink_img(img)

    cv2.imshow(WINDOW_NAME, img)
    cv2.waitKey(20)
