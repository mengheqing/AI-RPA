from typing import List


def turn_to_xywh(data):
    """
    输入为paddleocr识别的bbox格式，输出为xywh格式
    :param data:
    :return:
    """
    x, y, w, h = data[0][0], data[0][1], data[2][0] - data[0][0], data[2][1] - data[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h


def turn_to_xyxy(data):
    """
    输入为paddleocr识别的bbox格式，输出为xyxy格式
    :param data:
    :return:
    """
    x, y, xp, yp = data[0][0], data[0][1], data[2][0], data[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp


def turn_to_xywh_yolo(data):
    """
    输入为yolo识别的bbox格式，输出为xywh格式
    :param data:
    :return:
    """
    x, y, w, h = data[0], data[1], data[2] - data[0], data[3] - data[1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h


def int_box_area(box, w, h):
    """
    输入为xywh格式的bbox，输出为int类型的bbox面积
    :param box:
    :param w:
    :param h:
    :return:
    """
    x1, y1, x2, y2 = box
    int_box = [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]
    area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])
    return area


def remove_overlap_new(boxes, iou_threshold, ocr_bbox=None):
    '''
    ocr_bbox format: [{'type': 'text', 'bbox':[x,y], 'interactivity':False, 'content':str }, ...]
    boxes format: [{'type': 'icon', 'bbox':[x,y], 'interactivity':True, 'content':None }, ...]

    '''
    assert ocr_bbox is None or isinstance(ocr_bbox, List)