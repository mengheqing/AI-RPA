

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