import time
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


def int_box_middle(box, w, h):
    """
    输入为whwh格式的bbox，输出为int类型的bbox中心点坐标
    :param box:
    :return:
    """
    x1, y1, x2, y2 = box
    int_box = [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]
    middle = [round((int_box[0] + int_box[2]) / 4), round((int_box[1] + int_box[3]) / 4)]
    return middle


def remove_overlap_new(boxes, iou_threshold, ocr_bbox=None):
    '''
    ocr_bbox format: [{'type': 'text', 'bbox':[x,y], 'interactivity':False, 'content':str }, ...]
    boxes format: [{'type': 'icon', 'bbox':[x,y], 'interactivity':True, 'content':None }, ...]
    '''
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        # return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.80

    # boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1_elem in enumerate(boxes):
        box1 = box1_elem['bbox']
        is_valid_box = True
        for j, box2_elem in enumerate(boxes):
            # keep the smaller box
            box2 = box2_elem['bbox']
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            if ocr_bbox:
                # keep yolo boxes + prioritize ocr label
                box_added = False
                ocr_labels = ''
                for box3_elem in ocr_bbox:
                    if not box_added:
                        box3 = box3_elem['bbox']
                        if is_inside(box3, box1): # ocr inside icon
                            # box_added = True
                            # delete the box3_elem from ocr_bbox
                            try:
                                # gather all ocr labels
                                ocr_labels += box3_elem['content'] + ' '
                                filtered_boxes.remove(box3_elem)
                            except:
                                continue
                            # break
                        elif is_inside(box1, box3): # icon inside ocr, don't added this icon box, no need to check other ocr bbox bc no overlap between ocr bbox, icon can only be in one ocr box
                            box_added = True
                            break
                        else:
                            continue
                if not box_added:
                    if ocr_labels:
                        filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': ocr_labels, 'source':'box_yolo_content_ocr'})
                    else:
                        filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': None, 'source':'box_yolo_content_yolo'})
            else:
                filtered_boxes.append(box1)
    # torch.tensor(filtered_boxes)
    return filtered_boxes


def send_recognition_request(server_ip, image_path, image_name):
    """
    向模型服务器发送请求，获取识别结果
    :param server_ip: 模型服务器的IP地址
    :param image_path: 图像文件的路径
    :param image_name: 图像文件的名称
    :return: 识别结果
    """
    import requests
    from io import BytesIO
    from PIL import Image
    import base64
    url = f"http://{server_ip}:10088/api/element_recognition/image_recognition/get_image_recognition_result"

    payload = {}
    files = [
        ('image', ('input.png', open(image_path + 'model_decision/data/' + image_name, 'rb'),
                   'image/png'))
    ]
    headers = {}
    flag = False

    for _ in range(3):
        response = (requests.request("POST", url, headers=headers, data=payload, files=files))
        try:
            response = response.json()
            flag = True
            time.sleep(3)
            break
        except Exception as e:
            time.sleep(3)
            continue

    if not flag:
        print('response not json')
        response = response.json()

    # dict_keys(['label_coordinates', 'labeled_image', 'parsed_content', 'status'])
    label_coordinates = response['label_coordinates']
    labeled_image = response['labeled_image']
    parsed_content = response['parsed_content']
    status = response['status']
    # base64_to_image(labeled_image, path + 'model_decision/data/output.png')

    # 创建BytesIO对象并打开图片
    if ',' in labeled_image:
        header, base64_data = labeled_image.split(',', 1)
    else:
        base64_data = labeled_image

    # 解码Base64字符串
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data))
    # 获取图片尺寸
    width, height = image.size

    """
    {
    'bbox': [0.08749999850988388, 0.044285714626312256, 0.7095237970352173, 0.07095237821340561], 
    'content': 'ecs-workbench.aliyun.com/?from=ecs&instanceType=ecs&regionld=us-east-1&instanceld=i-0xj33dgimnjtvm2wlh5&resourceGroupld=&language=zh-CN ', 
    'interactivity': False, 
    'source': 'box_ocr_content_ocr', 
    'type': 'text'
    }
    
    # 创建BytesIO对象并打开图片
    image = Image.open(BytesIO(labeled_image))
    # 获取图片尺寸
    width, height = image.size
    # 保存标记图片至本地
    base64_to_image(labeled_image, get_legal_path(image_path) + 'model_decision/data/output.png')
    # 删除本地文件
    os.remove(get_legal_path(image_path) + 'model_decision/data/' + image_name)
    # 遍历结果列表，将bbox中的坐标值乘以图片尺寸
    # [v[0] / w, v[1] / h, v[2] / w, v[3] / h
    for item in result:
        item['bbox'] = [item['bbox'][0] * width, item['bbox'][1] * height, item['bbox'][2] * width, item['bbox'][3] * height]
    
    """
    result = {}
    count = 1
    for item in parsed_content.split('\n'):
        item = eval(item.split(':', 1)[1].strip())
        result[count] = {
            'type': item['type'],
            'middle_coordinate': int_box_middle(item['bbox'], width, height),
            'interactivity': item['interactivity'],
            'content': item['content'],
            'source': item['source']
        }
        count += 1

    return result, labeled_image
