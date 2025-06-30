from execution.base_tool import BaseTool, ToolResult
from typing import Any, Dict, List, Optional
from function import get_legal_path
import subprocess
import psutil
import time

class ScreenCapture(BaseTool):
    """
    截取屏幕图像。做法为使用Pillow的ImageGrab.grab()函数截取整个屏幕，并将截图保存至指定的项目路径中。
    工具的入参：image_path，保存截图的路径。
    工具的出参：image_name，保存截图的文件名称。
    """
    name: str = "screen_capture"
    description: str = """
    截取屏幕图像。做法为使用Pillow的ImageGrab.grab()函数截取整个屏幕，并将截图保存至指定的项目路径中。
    工具的入参：image_path，保存截图的路径。
    工具的出参：image_name，保存截图的文件名称。
    """
    parameters: dict = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "保存截图的路径。指定为本项目的项目路径。",
                "default": './',
            },
        },
        "required": ["image_path"],
    }

    def execute(
            self,
            image_path: Optional[str] = './'
    ) -> ToolResult:
        try:
            image_name = self.capture_screen(image_path)
            return ToolResult(output=image_name)
        except Exception as e:
            return ToolResult(error=str(e))

    def capture_screen(self, image_path: str) -> str:
        """
        截取屏幕图像并保存
        :param image_path: 保存截图的路径
        :return: 保存截图的文件名称
        """
        from PIL import ImageGrab
        import uuid
        # 截取屏幕
        screenshot = ImageGrab.grab()
        # 生成唯一的文件名
        image_name = f"{uuid.uuid4()}.png"
        # 保存截图
        screenshot.save(f"{get_legal_path(image_path)}/model_decision/data/{image_name}")
        return image_name


class CommandTool(BaseTool):
    """
    执行一个linux系统命令。做法为执行subprocess.run(command, shell=True, capture_output=True, text=True)函数。
    其中command为要求执行的系统命令。
    工具将捕获subprocess.run函数的输出并返回。
    """

    name: str = "system_command"
    description: str = """
    执行一个linux系统命令。做法为执行subprocess.run(command, shell=True, capture_output=True, text=True)函数。
    其中command为要求执行的系统命令。
    工具将捕获subprocess.run函数的输出并返回。
    """
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "可以直接执行的系统命令（如 'echo Hello World' 等），禁止包含用户输入的未过滤参数。",
                "default": '',
            },
        },
        "required": ["command"],
    }

    def execute(
            self,
            command: Optional[str] = ''
    ) -> ToolResult:
        try:
            result = self.run_command(command)
            # print(f'命令"{command} {args} {env}"执行成功: {result}')
            # return ToolResult(output=f'命令"{command} {args} {env}"执行成功: {result}')
            return ToolResult(output=result)
        except Exception as e:
            # print(f'命令"{command} {args} {env}"执行失败: {str(e)}')
            return ToolResult(error=str(e))


    def run_command(self, command: str):
        result = subprocess.run(
            command,  # 命令和参数分开为列表
            shell=True,  # 使用shell执行命令
            capture_output=True,  # 捕获输出
            text=True  # 返回字符串而非字节
        )
        time.sleep(3)
        if result.returncode == 0:
            return result.stdout
        else:
            raise Exception(result.stderr)


class OpenMacAppTool(BaseTool):
    """
    通过Mac电脑的应用程序名称打开一个应用，做法为subprocess.run(["open", "-a", "AppName"])，其中AppName为用户输入的应用名称。
    """

    name: str = "open_mac_app"
    description: str = """
    通过Mac电脑的应用程序名称打开一个应用，做法为subprocess.run(["open", "-a", "AppName"])，其中AppName为用户输入的应用名称。
    """
    parameters: dict = {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "(optional)打开的应用名称",
                "default": 'WeChat',
            }
        },
        "required": ["app_name"],
    }

    def execute(
            self,
            app_name: Optional[str] = None
    ) -> ToolResult:
        if not app_name:
            app_name = 'WeChat'
        try:
            self.open_app(app_name)
            return ToolResult(output=f'应用{app_name}打开成功')
        except Exception as e:
            return ToolResult(error=f'应用{app_name}打开失败: {str(e)}')


    def open_app(self, app_name: str):
        # 检查微信是否已在运行
        for proc in psutil.process_iter(["name"]):
            if proc.info["name"] == app_name:
                print("应用已在运行")
                return

        # 如果未运行，则打开微信
        result = subprocess.run(["open", "-a", app_name], capture_output=True, text=True)
        time.sleep(3)
        if result.returncode == 0:
            print("应用已成功打开")
        else:
            raise Exception(result.stderr)


class MouseTool(BaseTool):
    """
    执行鼠标的各种操作。包括获取当前鼠标位置、移动鼠标、点击鼠标、拖动鼠标、滚动鼠标等。
    工具的入参：
    1、mouse_operation_type：鼠标要执行的操作。如：
        1.1、get_position：获取当前鼠标位置。
        1.2、move_to：移动鼠标到指定位置。
        1.3、click：点击鼠标。
        1.4、drag：拖动鼠标。
        1.5、scroll：滚动鼠标。
    2、mouse_operation_args：鼠标要执行的操作的参数。如：
        2.1、get_position：无参数。
        2.2、move_to：移动鼠标到指定位置的参数。如：{"x": 100, "y": 100}。
        2.3、click：点击鼠标的参数。如：{"button": "left", "clicks": 1}。
        2.4、drag：拖动鼠标的参数。如：{"x": 100, "y": 100, "duration": 1}。
        2.5、scroll：滚动鼠标的参数。如：{"x": 0, "y": 100}。
    工具的出参：工具调用结束后，鼠标的位置。
    """
    name: str = "mouse_operation"
    description: str = """
    执行鼠标的各种操作。包括获取当前鼠标位置、移动鼠标、点击鼠标、拖动鼠标、滚动鼠标等。
    工具的入参：
    1、mouse_operation_type：鼠标要执行的操作。如：
        1.1、get_position：获取当前鼠标位置。
        1.2、move_to：移动鼠标到指定位置。
        1.3、click：点击鼠标。
        1.4、move_to_and_click：移动鼠标到指定位置并点击鼠标。
        1.5、drag：拖动鼠标。
        1.6、scroll：滚动鼠标。
    2、mouse_operation_args：鼠标要执行的操作的参数。如：
        2.1、get_position：无参数。
        2.2、move_to：移动鼠标到指定位置的参数。如：{"x": 100, "y": 100}。
        2.3、click：点击鼠标的参数。如：{"button": "left", "clicks": 1}。
        2.4、move_to_and_click：移动鼠标到指定位置并点击鼠标的参数。如：{"x": 100, "y": 100, "button": "left", "clicks": 1}。
        2.5、drag：拖动鼠标的参数。如：{"x": 100, "y": 100, "duration": 1}。
        2.6、scroll：滚动鼠标的参数。如：{"x": 0, "y": 100}。
    工具的出参：工具调用结束后，鼠标的位置。
    """
    parameters: dict = {
        "type": "object",
        "properties": {
            "mouse_operation_type": {
                "type": "string",
                "description": "鼠标要执行的操作。如：get_position、move_to、click、move_to_and_click、drag、scroll。",
                "enum": ["get_position", "move_to", "click", "move_to_and_click", "drag", "scroll"],
            },
            "mouse_operation_args": {
                "type": "object",
                "description": "鼠标要执行的操作的参数。",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "鼠标要移动到的x坐标。",
                    },
                    "y": {
                        "type": "integer",
                        "description": "鼠标要移动到的y坐标。",
                    },
                    "button": {
                        "type": "string",
                        "description": "鼠标要点击的按钮。",
                        "enum": ["left", "right", "middle"],
                    },
                    "clicks": {
                        "type": "integer",
                        "description": "鼠标要点击的次数。",
                    },
                    "duration": {
                        "type": "number",
                        "description": "鼠标拖动的持续时间。",
                    },
                },
            },
        },
        "required": ["mouse_operation_type"],
    }

    def execute(
            self,
            mouse_operation_type: str,
            mouse_operation_args: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        try:
            position = self.perform_operation(mouse_operation_type, mouse_operation_args)
            return ToolResult(output=f'鼠标操作{mouse_operation_type}成功。位置：{str(position)}')
        except Exception as e:
            return ToolResult(error=f'鼠标操作{mouse_operation_type}失败: {str(e)}')

    def perform_operation(self, mouse_operation_type, mouse_operation_args):
        import pyautogui
        import time
        if mouse_operation_type == "get_position":
            position = pyautogui.position()
            print(f"当前鼠标位置: {position}")
        elif mouse_operation_type == "move_to":
            x = mouse_operation_args.get("x", 0)
            y = mouse_operation_args.get("y", 0)
            pyautogui.moveTo(x, y, duration=1)
            print(f"鼠标移动到: ({x}, {y})")
        elif mouse_operation_type == "click":
            button = mouse_operation_args.get("button", "left")
            clicks = mouse_operation_args.get("clicks", 1)
            pyautogui.click(button=button, clicks=clicks)
            print(f"鼠标点击: {button} {clicks} 次")
        elif mouse_operation_type == "drag":
            x = mouse_operation_args.get("x", 0)
            y = mouse_operation_args.get("y", 0)
            duration = mouse_operation_args.get("duration", 1)
            pyautogui.dragTo(x, y, duration=duration)
            print(f"鼠标拖动到: ({x}, {y})，持续时间: {duration} 秒")
        elif mouse_operation_type == "scroll":
            x = mouse_operation_args.get("x", 0)
            y = mouse_operation_args.get("y", 0)
            pyautogui.scroll(x, y)
            print(f"鼠标滚动: ({x}, {y})")
        elif mouse_operation_type == "move_to_and_click":
            x = mouse_operation_args.get("x", 0)
            y = mouse_operation_args.get("y", 0)
            button = mouse_operation_args.get("button", "left")
            clicks = mouse_operation_args.get("clicks", 1)
            pyautogui.moveTo(x, y, duration=1)
            pyautogui.click(button=button, clicks=clicks)
        else:
            raise ValueError(f"未知的鼠标操作类型: {mouse_operation_type}")
        time.sleep(1)
        return pyautogui.position()


class KeyboardTool(BaseTool):
    """
    执行键盘的各种操作。包括输入字符串、按下和释放单个键、组合键。
    键盘的特殊键选项有：
    "enter", "esc", "backspace", "tab", "space",
    "shift", "ctrl", "alt", "win", "capslock",
    "home", "end", "pageup", "pagedown",
    "insert", "delete", "left", "right", "up", "down"
    工具的入参：
    1、keyboard_operation_type：键盘要执行的操作。如：
        1.1 输入字符串：input_string。
        1.2 按下和释放单个键：press_and_release_key。
        1.3 组合键：press_and_release_combo。
    2、keyboard_operation_args：键盘要执行的操作的参数。如：
        2.1 输入字符串：输入的字符串。
        2.2 按下和释放单个键：按下和释放的键。
        2.3 组合键：组合键的键列表。
    工具的出参：成功输入的键。
    """
    name: str = "keyboard_operation"
    description: str = """
    执行键盘的各种操作。包括输入字符串、按下和释放单个键、组合键。
    键盘的特殊键选项有：
    "enter", "esc", "backspace", "tab", "space", 
    "shift", "ctrl", "alt", "win", "capslock", 
    "home", "end", "pageup", "pagedown", 
    "insert", "delete", "left", "right", "up", "down"
    工具的入参：
    1、keyboard_operation_type：键盘要执行的操作。如：
        1.1 输入字符串：input_string。
        1.2 按下和释放单个键：press_and_release_key。
        1.3 组合键：press_and_release_combo。
    2、keyboard_operation_args：键盘要执行的操作的参数。如：
        2.1 输入字符串：输入的字符串。
        2.2 按下和释放单个键：按下和释放的键。
        2.3 组合键：组合键的键列表。
    工具的出参：成功输入的键。
    """
    parameters: dict = {
        "type": "object",
        "properties": {
            "keyboard_operation_type": {
                "type": "string",
                "description": "键盘要执行的操作。如：input_string、press_and_release_key、press_and_release_combo。",
                "enum": ["input_string", "press_and_release_key", "press_and_release_combo"],
            },
            "keyboard_operation_args": {
                "type": "object",
                "description": "键盘要执行的操作的参数。",
                "properties": {
                    "string": {
                        "type": "string",
                        "description": "输入的字符串。",
                    },
                    "key": {
                        "type": "string",
                        "description": "按下和释放的键。",
                    },
                    "combo": {
                        "type": "array",
                        "description": "组合键的键列表。",
                        "items": {
                            "type": "string",
                        },
                    },
                },
            },
        },
        "required": ["keyboard_operation_type"],
    }

    def execute(
            self,
            keyboard_operation_type: str,
            keyboard_operation_args: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        try:
            result = self.perform_operation(keyboard_operation_type, keyboard_operation_args)
            return ToolResult(output=f'键盘操作{keyboard_operation_type}成功。输入的键：{str(result)}')
        except Exception as e:
            return ToolResult(error=f'键盘操作{keyboard_operation_type}失败: {str(e)}')

    def perform_operation(self, keyboard_operation_type, keyboard_operation_args):
        import pyautogui
        import time
        if keyboard_operation_type == "input_string":
            string = keyboard_operation_args.get("string", "")
            pyautogui.write(string, interval=0.2)
            print(f"输入字符串: {string}")
        elif keyboard_operation_type == "press_and_release_key":
            key = keyboard_operation_args.get("key", "")
            pyautogui.press(key)
            print(f"按下和释放键: {key}")
        elif keyboard_operation_type == "press_and_release_combo":
            combo = keyboard_operation_args.get("combo", [])
            pyautogui.hotkey(*combo)
            print(f"组合键: {combo}")
        else:
            raise ValueError(f"未知的键盘操作类型: {keyboard_operation_type}")

        time.sleep(1)
        return str(keyboard_operation_args)



