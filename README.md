# OpenLabeling: open-source image and video labeler

[![GitHub stars](https://img.shields.io/github/stars/Cartucho/OpenLabeling.svg?style=social&label=Stars)](https://github.com/Cartucho/OpenLabeling)

Image labeling in multiple annotation formats:
- PASCAL VOC (= [darkflow](https://github.com/thtrieu/darkflow))
- [YOLO darknet](https://github.com/pjreddie/darknet)
- ask for more (create a new issue)...

<img src="https://media.giphy.com/media/l49JDgDSygJN369vW/giphy.gif" width="40%"><img src="https://media.giphy.com/media/3ohc1csRs9PoDgCeuk/giphy.gif" width="40%">
<img src="https://media.giphy.com/media/3o752fXKwTJJkhXP32/giphy.gif" width="40%"><img src="https://media.giphy.com/media/3ohc11t9auzSo6fwLS/giphy.gif" width="40%">

## Latest Features

- Jun 2019: Deep Learning Object Detection Model
- May 2019: [ECCV2018] Distractor-aware Siamese Networks for Visual Object Tracking
- Jan 2019: easy and quick bounding-boxe's resizing!
- Jan 2019: video object tracking with OpenCV trackers!
- TODO: Label photos via Google drive to allow "team online labeling".
[New Features Discussion](https://github.com/Cartucho/OpenLabeling/issues/3)

## Table of contents

- [Quick start](#quick-start)
- [Prerequisites](#prerequisites)
- [Run project](#run-project)
- [GUI usage](#gui-usage)
- [Authors](#authors)

## Quick start

To start using the YOLO Bounding Box Tool you need to [download the latest release](https://github.com/Cartucho/OpenLabeling/archive/v1.3.zip) or clone the repo:

```
git clone https://github.com/Cartucho/OpenLabeling
```

### Prerequisites

You need to install:

- [Python](https://www.python.org/downloads/)
- [OpenCV](https://opencv.org/) version >= 3.0
    1. `python -mpip install -U pip`
    1. `python -mpip install -U opencv-python`
    1. `python -mpip install -U opencv-contrib-python`
- numpy, tqdm and lxml:
    1. `python -mpip install -U numpy`
    1. `python -mpip install -U tqdm`
    1. `python -mpip install -U lxml`

Alternatively, you can install everything at once by simply running:

```
python -mpip install -U pip
python -mpip install -U -r requirements.txt
```

### Run project

Step by step:

  1. Open the `main/` directory
  2. Insert the input images and videos in the folder **input/**
  3. Insert the classes in the file **class_list.txt** (one class name per line)
  4. Run the code:

         python main.py [-h] [-i] [-o] [-t]

         optional arguments:
          -h, --help                Show this help message and exit
          -i, --input               Path to images and videos input folder | Default: input/
          -o, --output              Path to output folder (if using the PASCAL VOC format it's important to set this path correctly) | Default: output/
          -t, --thickness           Bounding box and cross line thickness (int) | Default: -t 1

  5. You can find the annotations in the folder **output/**

#### How to use the deep learning feature

- Download one or some deep learning models from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
  and put it into `object_detection/models` directory (you need to create the `models` folder by yourself). The outline of `object_detection` looks like that:
  + `tf_object_detection.py`
  + `utils.py`
  + `models/ssdlite_mobilenet_v2_coco_2018_05_09`

Download the pre-trained model by clicking this link http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz and put it into `object_detection/models`. Create the `models` folder if necessary. Make sure to extract the model.

  **Note**: Default model used in `main_auto.py` is `ssdlite_mobilenet_v2_coco_2018_05_09`. We can
  set `graph_model_path` in file `main_auto.py` to change the pretrain model
- Using `main_auto.py` to automatically label data first

  TODO: explain how the user can 

### GUI usage

Keyboard, press: 

<img src="https://github.com/Cartucho/OpenLabeling/blob/master/keyboard_usage.jpg">

| Key | Description |
| --- | --- |
| a/d | previous/next image |
| s/w | previous/next class |
| e | edges |
| h | help |
| q | quit |

Video:

| Key | Description |
| --- | --- |
| p | predict the next frames' labels |

Mouse:
  - Use two separate left clicks to do each bounding box
  - **Right-click** -> **quick delete**!
  - Use the middle mouse to zoom in and out
  - Use double click to select a bounding box

## Authors

* **João Cartucho** - Please give me your feedback: to.cartucho@gmail.com

    Feel free to contribute

    [![GitHub contributors](https://img.shields.io/github/contributors/Cartucho/OpenLabeling.svg)](https://github.com/Cartucho/OpenLabeling/graphs/contributors)
