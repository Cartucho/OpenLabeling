# YOLO v2 Bounding Box Tool

[![New](https://img.shields.io/badge/2018-NEW-brightgreen.svg)](https://github.com/Cartucho/yolo-boundingbox-labeler-GUI/commits/master)
[![GitHub stars](https://img.shields.io/github/stars/Cartucho/yolo-boundingbox-labeler-GUI.svg?style=social&label=Stars)](https://github.com/Cartucho/yolo-boundingbox-labeler-GUI)

Bounding box labeler tool to generate the training data in the format YOLO v2 requires.

<img src="https://media.giphy.com/media/l49JDgDSygJN369vW/giphy.gif" width="40%"><img src="https://media.giphy.com/media/3ohc1csRs9PoDgCeuk/giphy.gif" width="40%">
<img src="https://media.giphy.com/media/3o752fXKwTJJkhXP32/giphy.gif" width="40%"><img src="https://media.giphy.com/media/3ohc11t9auzSo6fwLS/giphy.gif" width="40%">

The idea is to use OpenCV so that later it uses SIFT and Tracking algorithms to make labeling easier.

I wanted this tool to give automatic suggestions for the labels!
[New Features Discussion](https://github.com/Cartucho/yolo-boundingbox-labeler-GUI/issues/3)

## Table of contents

- [Quick start](#quick-start)
- [Prerequisites](#prerequisites)
- [Run project](#run-project)
- [GUI usage](#gui-usage)
- [Authors](#authors)

## Quick start

To start using the YOLO Bounding Box Tool you need to [download the latest release](https://github.com/Cartucho/yolo-boundingbox-labeler-GUI/archive/v1.0.zip) or clone the repo:

```
git clone https://github.com/Cartucho/yolo-boundingbox-labeler-GUI
```

### Prerequisites

You need to install:

- [OpenCV](https://opencv.org/) version >= 3.0
  - Installation in Windows: `pip install opencv-python`
  - [Installation in Linux](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)
- [Python](https://www.python.org/downloads/)

### Run project

Step by step:

  1. Insert the images in the folder **images/**
  2. Insert the class list in the file **class_list.txt**
  3. Run the code:
         ```
         python run.py
         ```
  4. You can find the bounding box files in the folder **bbox_txt/**

### GUI usage

Keyboard, press: 

<img src="https://github.com/Cartucho/yolo-boundingbox-labeler-GUI/blob/master/keyboard_usage.jpg">

| Key | Description |
| --- | --- |
| h | help |
| q | quit |
| e | edges |
| a/d | previous/next image |
| s/w | previous/next class |


Mouse:
  - Use two left clicks to do each bounding box
  - Use the middle mouse to zoom in and out

## Authors

* **João Cartucho** - Please give me your feedback: to.cartucho@gmail.com

    Feel free to contribute

    [![GitHub contributors](https://img.shields.io/github/contributors/Cartucho/yolo-boundingbox-labeler-GUI.svg)](https://github.com/Cartucho/yolo-boundingbox-labeler-GUI/graphs/contributors)
