from execution.base_tool import BaseTool, ToolResult
from config import model_server_ip
from model_decision.function.functions import send_recognition_request
from element_recognition.function.functions import base64_to_image
from function import get_legal_path

class ImageRecognition(BaseTool):
    """
    此工具用于识别当前页面的图像元素，包括文本、图标等，并返回结构化的识别结果。
    输入为：
    1、当前项目目录;
    2、当前设备桌面的截图文件名称。
    输出为格式化的识别结果，格式如下：
    [
        {
            'middle_coordinate': [500, 500],
            'content': '登录',
            'interactivity': True,
            'source': 'box_ocr_content_ocr',
            'type': 'text'
        }
    ]
    其中，middle_coordinate表示元素的中心点坐标，content表示元素的文本内容，interactivity表示元素是否可交互，source表示元素的识别来源，type表示元素的类型。
    """
    name: str = "image_recognition"
    description: str = """
    此工具用于识别当前页面的图像元素，包括文本、图标等，并返回结构化的识别结果。
    输入为：
    1、当前项目目录;
    2、当前设备桌面的截图文件名称。
    输出为格式化的识别结果，格式如下：
    "[
        {
            'middle_coordinate': [500, 500],
            'content': '登录', 
            'interactivity': True, 
            'source': 'box_ocr_content_ocr', 
            'type': 'text'
        }
    ]"
    其中，middle_coordinate表示元素的中心点坐标，content表示元素的文本内容，interactivity表示元素是否可交互，source表示元素的识别来源，type表示元素的类型。
    """
    parameters: dict = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "当前项目目录。",
            },
            "image_name": {
                "type": "string",
                "description": "当前设备桌面的截图文件名称。",
            }
        },
        "required": ["image_name", "image_path"],
    }

    def execute(self, image_path: str, image_name: str) -> ToolResult:
        """向模型服务器发送请求，获取识别结果"""
        import json
        import os
        result, labeled_image = send_recognition_request(model_server_ip, get_legal_path(image_path), image_name)
        # 保存标记图片至本地
        base64_to_image(labeled_image, get_legal_path(image_path) + 'model_decision/data/output.png')
        # 删除本地文件
        os.remove(get_legal_path(image_path) + 'model_decision/data/' + image_name)
        return ToolResult(output=json.dumps(result))
