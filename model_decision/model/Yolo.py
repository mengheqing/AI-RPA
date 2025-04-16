from ultralytics import YOLO
# Load the model.


def load_yolo_model(model_path):
    """
    加载yolo模型
    :param model_path:
    :return:
    """
    model = YOLO(model_path)
    return model
