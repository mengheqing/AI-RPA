from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTool(ABC, BaseModel):
    """
        抽象基类，代表一个基础工具，继承自 pydantic 的 BaseModel，用于定义工具的基本结构。
        """
    name: str  # 工具的名称
    description: str  # 工具的描述信息
    parameters: Optional[dict] = None  # 工具的参数，可选

    class Config:
        arbitrary_types_allowed = True  # 允许使用任意类型

    def __call__(self, **kwargs) -> Any:
        """
        使类实例可调用，调用时执行工具的 execute 方法。

        Args:
            **kwargs: 传递给工具执行方法的关键字参数。

        Returns:
            Any: 工具执行后的结果。
        """
        return self.execute(**kwargs)

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        抽象方法，子类必须实现该方法以定义工具的具体执行逻辑。

        Args:
            **kwargs: 执行工具所需的关键字参数。

        Returns:
            Any: 工具执行后的结果。
        """

    def to_param(self) -> Dict:
        """
        将工具转换为函数调用所需的参数格式。

        Returns:
            Dict: 包含工具信息的字典，符合函数调用的格式要求。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolResult(BaseModel):
    """
    表示工具执行的结果。
    """
    # 工具执行的输出结果
    output: Any = Field(default=None)
    # 工具执行过程中出现的错误信息，可选
    error: Optional[str] = Field(default=None)
    # 以 base64 编码的图片数据，可选
    base64_image: Optional[str] = Field(default=None)
    # 系统相关信息，可选
    system: Optional[str] = Field(default=None)

    class Config:
        # 允许使用任意类型
        arbitrary_types_allowed = True

    def __bool__(self):
        """
        判断工具执行结果是否包含有效数据。

        Returns:
            bool: 如果任意字段有值则返回 True，否则返回 False。
        """
        return any(getattr(self, field) for field in self.__fields__)

    def __add__(self, other: "ToolResult"):
        """
        将当前工具执行结果与另一个工具执行结果合并。

        Args:
            other (ToolResult): 另一个工具执行结果实例。

        Returns:
            ToolResult: 合并后的工具执行结果实例。

        Raises:
            ValueError: 当某些字段不允许合并时抛出。
        """
        def combine_fields(
            field: Optional[str], other_field: Optional[str], concatenate: bool = True
        ):
            """
            合并两个字段的值。

            Args:
                field (Optional[str]): 当前实例的字段值。
                other_field (Optional[str]): 另一个实例的字段值。
                concatenate (bool, optional): 是否拼接两个字段的值。默认为 True。

            Returns:
                Optional[str]: 合并后的字段值。

            Raises:
                ValueError: 当 concatenate 为 False 且两个字段都有值时抛出。
            """
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("无法合并工具执行结果")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def __str__(self):
        """
        将工具执行结果转换为字符串表示。

        Returns:
            str: 如果存在错误信息则返回错误信息，否则返回输出结果。
        """
        return f"错误信息: {self.error}" if self.error else self.output

    def replace(self, **kwargs):
        """
        返回一个新的 ToolResult 实例，其中指定字段被替换为新值。

        Args:
            **kwargs: 要替换的字段及其新值。

        Returns:
            ToolResult: 包含新字段值的新 ToolResult 实例。
        """
        # return self.copy(update=kwargs)
        return type(self)(**{**self.dict(), **kwargs})

class CLIResult(ToolResult):
    """
    一个可以渲染为命令行输出的工具执行结果类。
    """

class ToolFailure(ToolResult):
    """
    表示工具执行失败的工具执行结果类。
    """