import os

from element_recognition.element_recognition_server import app
from flask import request, jsonify
import torch
from PIL import Image
from model_decision.util.vision_model_utils import check_ocr_box, get_caption_model_processor, get_som_labeled_img
from model_decision.model.Yolo import load_yolo_model
from config import project_path
from function import get_legal_path
from datetime import datetime

path = get_legal_path(project_path)
yolo_model = load_yolo_model(model_path=path + 'model_decision/weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path=path + "model_decision/weights/icon_caption_florence"
)

def process_image(path, image):
    # 初始化模型
    # print('初始化模型')
    # yolo_model = load_yolo_model(model_path=path + 'model_decision/weights/icon_detect/model.pt')
    # caption_model_processor = get_caption_model_processor(
    #     model_name="florence2",
    #     model_name_or_path=path + "model_decision/weights/icon_caption_florence"
    # )
    #
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('初始化完成')

    box_threshold: float = 0.05
    iou_threshold: float = 0.1

    imgsz: int = 640

    # # 读取图片
    # print('读取图片')
    # image_path = path + 'model_decision/data/input.png'
    # image = Image.open(image_path).convert('RGB')

    # 临时保存图片
    print('临时保存图片')
    image_save_path = path + 'model_decision/data/temp_image' + str(int(datetime.now().timestamp())) + '.png'
    image.save(image_save_path)

    # 配置绘制边界框的参数
    print('配置绘制边界框的参数')
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    # OCR处理
    print('OCR处理')
    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_save_path,
        display_img=False,
        output_bb_format='xyxy',
        goal_filtering=None,
        ocr_args={'paragraph': False, 'threshold': 0.9},
    )

    text, ocr_bbox, size = ocr_bbox_rslt

    # 获取标记后的图片和解析内容
    print('获取标记后的图片和解析内容')
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_save_path,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
    )

    # 格式化解析结果
    print('格式化解析结果')
    parsed_content = '\n'.join([f'icon {i}: {str(v)}' for i, v in enumerate(parsed_content_list)])

    # 删除临时文件
    os.remove(image_save_path)

    return {
        "status": "success",
        "labeled_image": dino_labled_img,  # base64编码的图片
        "parsed_content": parsed_content,
        "label_coordinates": label_coordinates
    }

@app.route('/api/element_recognition/image_recognition/get_image_recognition_result', methods=['POST'])
def get_image_recognition_result():
    # 检查请求中是否包含文件
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    file = request.files['image']
    # 检查文件是否有文件名
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        # try:
            # 使用 Pillow 打开图片
        img = Image.open(file.stream)
        img.verify()  # 验证图片完整性
        img = Image.open(file.stream)   # 重新打开，因为 verify 后流会关闭
        path = get_legal_path(project_path)
        result = process_image(path, img)
        return jsonify(result)
        # except Exception as e:
        #     return jsonify({"error": f"Failed to open image: {str(e)}"}), 400
    else:
        return jsonify({"error": "get file failed"}), 400