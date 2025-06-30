# import torch
# from PIL import Image
# import base64
# from config import project_path
from function import get_legal_path
# from model_decision.util.vision_model_utils import check_ocr_box, get_caption_model_processor, get_som_labeled_img
# from model_decision.model.Yolo import load_yolo_model
from element_recognition.function.functions import base64_to_image
import requests

# path = get_legal_path(project_path)
# yolo_model = load_yolo_model(model_path=path + 'model_decision/weights/icon_detect/model.pt')
# caption_model_processor = get_caption_model_processor(
#     model_name="florence2",
#     model_name_or_path=path + "model_decision/weights/icon_caption_florence"
# )

# def process_image(path):
#     # 初始化模型
#     print('初始化模型')
#     yolo_model = load_yolo_model(model_path=path + 'model_decision/weights/icon_detect/model.pt')
#     caption_model_processor = get_caption_model_processor(
#         model_name="florence2",
#         model_name_or_path=path + "model_decision/weights/icon_caption_florence"
#     )
#
#     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print('初始化完成')
#
#     box_threshold: float = 0.05
#     iou_threshold: float = 0.1
#
#     imgsz: int = 640
#
#     # 读取图片
#     print('读取图片')
#     image_path = path + 'model_decision/data/input.png'
#     image = Image.open(image_path).convert('RGB')
#
#     # 临时保存图片
#     print('临时保存图片')
#     image_save_path = path + 'model_decision/data/temp_image.png'
#     image.save(image_save_path)
#
#     # 配置绘制边界框的参数
#     print('配置绘制边界框的参数')
#     box_overlay_ratio = image.size[0] / 3200
#     draw_bbox_config = {
#         'text_scale': 0.8 * box_overlay_ratio,
#         'text_thickness': max(int(2 * box_overlay_ratio), 1),
#         'text_padding': max(int(3 * box_overlay_ratio), 1),
#         'thickness': max(int(3 * box_overlay_ratio), 1),
#     }
#
#     # OCR处理
#     print('OCR处理')
#     ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
#         image_save_path,
#         display_img=False,
#         output_bb_format='xyxy',
#         goal_filtering=None,
#         ocr_args={'paragraph': False, 'threshold': 0.9},
#     )
#
#     text, ocr_bbox, size = ocr_bbox_rslt
#
#     # 获取标记后的图片和解析内容
#     print('获取标记后的图片和解析内容')
#     dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
#         image_save_path,
#         yolo_model,
#         BOX_TRESHOLD=box_threshold,
#         output_coord_in_ratio=True,
#         ocr_bbox=ocr_bbox,
#         draw_bbox_config=draw_bbox_config,
#         caption_model_processor=caption_model_processor,
#         ocr_text=text,
#         iou_threshold=iou_threshold,
#         imgsz=imgsz,
#     )
#
#     # 格式化解析结果
#     print('格式化解析结果')
#     parsed_content = '\n'.join([f'icon {i}: {str(v)}' for i, v in enumerate(parsed_content_list)])
#
#     def base64_to_image(base64_string, output_path):
#         """
#         将Base64编码的字符串转换为图片并保存到本地。
#
#         :param base64_string: Base64编码的字符串（不包含头部信息）
#         :param output_path: 图片保存的路径（包括文件名和扩展名）
#         """
#         try:
#             # 去掉可能存在的头部信息（如"data:image/png;base64,"）
#             if ',' in base64_string:
#                 header, base64_data = base64_string.split(',', 1)
#             else:
#                 base64_data = base64_string
#
#             # 解码Base64字符串
#             image_data = base64.b64decode(base64_data)
#
#             # 将解码后的数据写入文件
#             with open(output_path, 'wb') as image_file:
#                 image_file.write(image_data)
#
#             print(f"图片已成功保存到 {output_path}")
#         except Exception as e:
#             print(f"发生错误：{e}")
#
#     base64_to_image(dino_labled_img, path + 'model_decision/data/output.jpg')
#
#     return {
#         "status": "success",
#         "labeled_image": dino_labled_img,  # base64编码的图片
#         "parsed_content": parsed_content,
#         "label_coordinates": label_coordinates
#     }


def try_request(path):
    url = "http://47.85.109.201:10088/api/element_recognition/image_recognition/get_image_recognition_result"

    payload = {}
    files = [
        ('image', ('input.png', open(path + 'model_decision/data/input.png', 'rb'),
                   'image/png'))
    ]
    headers = {}

    response = (requests.request("POST", url, headers=headers, data=payload, files=files))
    try:
        response = response.json()
    except Exception as e:
        print('response not json')
        print(response.text)
        raise e

    # dict_keys(['label_coordinates', 'labeled_image', 'parsed_content', 'status'])
    label_coordinates = response['label_coordinates']
    labeled_image = response['labeled_image']
    parsed_content = response['parsed_content']
    status = response['status']
    base64_to_image(labeled_image, path + 'model_decision/data/output.png')
    print(parsed_content)
    """
    {
    'bbox': [0.08749999850988388, 0.044285714626312256, 0.7095237970352173, 0.07095237821340561], 
    'content': 'ecs-workbench.aliyun.com/?from=ecs&instanceType=ecs&regionld=us-east-1&instanceld=i-0xj33dgimnjtvm2wlh5&resourceGroupld=&language=zh-CN ', 
    'interactivity': False, 
    'source': 'box_ocr_content_ocr', 
    'type': 'text'
    }
    """
    parsed_content.split('\n')[0].split(':', 1)[1].strip()


if __name__ == "__main__":
    project_path = '/Users/mengheqing/PycharmProjects/AI-RPA/'
    # project_path = '/root/AI-RPA/'
    path = get_legal_path(project_path)
    # result = process_image(path)
    # print(result)
    try_request(path)