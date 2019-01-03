# OpenLabeling - Open-source image labeling tool

[![GitHub stars](https://img.shields.io/github/stars/Cartucho/OpenLabeling.svg?style=social&label=Stars)](https://github.com/Cartucho/OpenLabeling)

Image labeling in multiple annotation formats:
- PASCAL VOC (= [darkflow](https://github.com/thtrieu/darkflow))
- [YOLO darknet](https://github.com/pjreddie/darknet)

<img src="https://media.giphy.com/media/l49JDgDSygJN369vW/giphy.gif" width="40%"><img src="https://media.giphy.com/media/3ohc1csRs9PoDgCeuk/giphy.gif" width="40%">
<img src="https://media.giphy.com/media/3o752fXKwTJJkhXP32/giphy.gif" width="40%"><img src="https://media.giphy.com/media/3ohc11t9auzSo6fwLS/giphy.gif" width="40%">

The idea is to use OpenCV so that later it uses SIFT and Tracking algorithms to make labeling easier.

I wanted this tool to give automatic suggestions for the labels!
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
    2. `python -mpip install -U opencv-python`
- numpy and tqdm:
    `python -mpip install -U numpy`
    `python -mpip install -U tqdm`

### Run project

Step by step:

  1. Insert the input images and videos in the folder **input/**
  2. Insert the classes in the file **class_list.txt** (one class name per line)
  3. Run the code:
         ```
         python main.py [-h] [-i] [-o] [-t]

         optional arguments:
          -h, --help                Show this help message and exit
          -i, --input               Path to images and videos input folder | Default: input/
          -o, --output              Path to output folder (if using the PASCAL VOC format it's important to set this path correctly) | Default: output/
          -t, --thickness           Bounding box and cross line thickness
         ```
  4. You can find the output files in the folder **output/**

### GUI usage

Keyboard, press: 

<img src="https://github.com/Cartucho/OpenLabeling/blob/master/keyboard_usage.jpg">

| Key | Description |
| --- | --- |
| h | help |
| q | quit |
| e | edges |
| a/d | previous/next image |
| s/w | previous/next class |


Mouse:
  - Use two separate left clicks to do each bounding box
  - Use the middle mouse to zoom in and out
  - Use double click to select a bounding box
  - Right click to quickly delete a bounding box

## Authors

* **João Cartucho** - Please give me your feedback: to.cartucho@gmail.com

    Feel free to contribute

    [![GitHub contributors](https://img.shields.io/github/contributors/Cartucho/OpenLabeling.svg)](https://github.com/Cartucho/OpenLabeling/graphs/contributors)
