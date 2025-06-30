def get_legal_path(path):
    path = path.replace('\\', '/')
    if path[-1] != '/':
        return path + '/'
    else:
        return path

def load_tools():
    import os
    import sys
    import importlib
    import inspect

    def get_classes_from_folder(folder_path):
        # 将目标文件夹添加到模块搜索路径
        sys.path.insert(0, folder_path)
        classes = []
        modules = {}
        # 遍历目标文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.py'):
                    # 去掉 .py 扩展名得到模块名
                    module_name = os.path.splitext(file)[0]
                    try:
                        # 动态导入模块
                        module = importlib.import_module(module_name)
                        # 获取模块中定义的所有类
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            # 确保类是在当前模块中定义的
                            if obj.__module__ == module_name:
                                classes.append(obj)
                                modules[obj.model_fields['name'].default] = obj
                    except ImportError:
                        continue
        # 从模块搜索路径中移除目标文件夹
        sys.path.pop(0)
        return classes, modules

    # 使用示例
    folder_path = 'execution/tools/'  # 替换为实际的文件夹路径
    all_classes, modules = get_classes_from_folder(folder_path)
    all_tools = []
    for cls in all_classes:
        name = cls.model_fields['name']
        description = cls.model_fields['description']
        parameters = cls.model_fields['parameters']
        all_tools.append(
            {
                'type': 'function',
                'function': {
                    'name': name.default,
                    'description': description.default,
                    'parameters': parameters.default
                }
            }
        )
    return all_tools, modules


def instantiate_and_call(tool_name):
    import importlib
    # 导入包含目标类的模块
    module = importlib.import_module('tools')

    # 遍历模块中的所有属性
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        # 检查是否为类，且类的 name 属性匹配
        if isinstance(attr, type) and hasattr(attr, 'name') and attr.name == tool_name:
            # 实例化类
            instance = attr()
            # 调用类的方法
            if hasattr(instance, 'execute'):
                result = instance.execute()
                return result
    return f"未找到名为 {tool_name} 的工具"


if __name__ == "__main__":
    load_tools()