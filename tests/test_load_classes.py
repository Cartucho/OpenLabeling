from main.load_classes import get_class_list


def test_get_class_list():
    classes = get_class_list()
    assert classes == ['person', 'billiard ball', 'donut']
