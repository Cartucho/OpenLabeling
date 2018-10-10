#------------------------------------------------------------------------------------------------
#
# By: Rafael Caballero Gonzalez
#
# Trying to predict labels in the next frames using OpenCV trackers
#
#------------------------------------------------------------------------------------------------

import os
import numpy as np
import cv2

class LabelTracker():
    def __init__(self, imgs_path_list, type='KCF'):

        if not self.is_opencv_version_ok():
            raise Exception('OpenCV version is under v3')

        self.imgs_path_list = imgs_path_list
        self.create_tmp_txt_files()

        self.tracker_type = type


    def predict_next_imgs(self, current_img_index, num_imgs_to_predict):
        current_img = cv2.imread(self.imgs_path_list[current_img_index])

        if len(self.imgs_path_list) > 1:
            next_imgs_path_list = self.imgs_path_list[current_img_index+1:current_img_index+num_imgs_to_predict]
            next_imgs = list()
            for next_img in next_imgs_path_list:
                next_imgs.append(cv2.imread(next_img))

            current_img_height, current_img_width, current_img_channels = current_img.shape

            classWithTracker = list()

            # load current bboxes
            txt_path = self.get_txt_path(self.imgs_path_list[current_img_index], 'bbox_txt/')
            with open(txt_path) as f:
                content = f.readlines()
            for line in content:
                values_str = line.split()
                class_index, x_center, y_center, x_width, y_height = map(float, values_str)
                roi = (x_center*current_img_width, y_center*current_img_height, x_width*current_img_width, y_height*current_img_height)
                tracker = cv2.TrackerKCF_create()
                tracker.init(current_img, roi)
                classWithTracker.append((class_index, tracker))

            # predict
            for i in range(0, len(next_imgs_path_list), 1):
                temp_txt_path = self.get_txt_path(next_imgs_path_list[i])
                with open(temp_txt_path, 'a') as f:
                    for class_index, tracker in classWithTracker:
                        ok, bbox = tracker.update(next_imgs[i])
                        if ok:
                            line = self.yolo_format(class_index, (bbox[0], bbox[1]),
                                                    (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                                    current_img_width, current_img_height)
                            f.write(line + "\n")

    def is_opencv_version_ok(self):
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(minor_ver) < 3:
            return False
        else:
            return True

    def create_tmp_txt_files(self):
        if not os.path.exists('tmp/'):
            os.makedirs('tmp/')

        for img_path in self.imgs_path_list:
            txt_path = self.get_txt_path(img_path)
            if not os.path.isfile(txt_path):
                open(txt_path, 'a').close()

    def get_txt_path(self, img_path, folder='tmp/'):
        img_name = os.path.basename(os.path.normpath(img_path))
        img_type = img_path.split('.')[-1]
        return folder + img_name.replace(img_type, 'txt')

    def yolo_format(self, class_index, point_1, point_2, width, height):
        x_center = (point_1[0] + point_2[0]) / float(2.0 * width)
        y_center = (point_1[1] + point_2[1]) / float(2.0 * height)
        x_width = float(abs(point_2[0] - point_1[0])) / width
        y_height = float(abs(point_2[1] - point_1[1])) / height
        return str(class_index) + " " + str(x_center) \
               + " " + str(y_center) + " " + str(x_width) + " " + str(y_height)

