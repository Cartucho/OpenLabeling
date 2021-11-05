import configparser

from pathlib import Path


def non_blank_lines(file_object):
    for l in file_object:
        line = l.rstrip()
        if line:
            yield line


def get_class_list():
    """
    Uses the most recent classes source file defined in config.ini
    otherwise defaults the example class_list.txt file provided in
    this repository.

    """
    config = configparser.ConfigParser()
    this_dir = Path(__file__).parent
    config_path = this_dir / "config.ini"
    config.read(str(config_path))
    most_recent_classes_file = config.get("CLASSES", "MOST_RECENT_FILE")

    if not most_recent_classes_file:
        classes_src = str(this_dir / "class_list.txt")
    else:
        classes_src = most_recent_classes_file

    with open(classes_src) as f:
        return list(non_blank_lines(file_object=f))
