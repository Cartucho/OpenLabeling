# OpenLabeling - Refactored

A modular image annotation tool for bounding box labeling, refactored for better maintainability and extensibility.

## Features

- **Modular Architecture**: Code organized into logical modules for easier maintenance
- **Type Hints**: Full type annotations for better code understanding and IDE support
- **Clean Code**: Improved readability and maintainability
- **Multiple Annotation Formats**: Support for both YOLO and PASCAL VOC formats
- **Video Tracking**: Support for tracking objects across video frames
- **Interactive UI**: Intuitive interface with mouse and keyboard controls

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m openlabeling.app --input_dir input --output_dir output --tracker KCF
```

Or after installing the package:

```bash
openlabeling --input_dir input --output_dir output --tracker KCF
```

## Command Line Options

- `-i, --input_dir`: Path to input directory (default: 'input')
- `-o, --output_dir`: Path to output directory (default: 'output')
- `-t, --thickness`: Bounding box and cross line thickness (default: 1)
- `--draw-from-PASCAL-files`: Draw bounding boxes from the PASCAL files (default: YOLO)
- `--tracker`: Tracker type to use (default: 'KCF')
- `-n, --n_frames`: Number of frames to track object for (default: 200)

## Controls

- `a/d`: Navigate between images
- `w/s`: Navigate between classes
- `Left Click`: Start/finish drawing a bounding box
- `Double Click`: Select a bounding box
- `Right Click`: Delete selected bounding box
- `e`: Toggle edge detection
- `p`: Start tracking selected objects in video
- `h`: Show help
- `q`: Quit

## Modules

- `app.py`: Main application logic
- `config.py`: Configuration and argument parsing
- `utils.py`: Utility functions
- `bbox_handler.py`: Bounding box operations
- `tracker.py`: Object tracking functionality

## Improvements Made

1. **Code Organization**: Split monolithic code into logical modules
2. **Type Safety**: Added comprehensive type hints throughout
3. **Documentation**: Added docstrings and comments for clarity
4. **Maintainability**: Reduced complexity by separating concerns
5. **Error Handling**: Improved error handling patterns
6. **Code Reusability**: Created reusable utility functions

## Original Project

This refactored version is based on the original OpenLabeling project with significant improvements to architecture and code quality.