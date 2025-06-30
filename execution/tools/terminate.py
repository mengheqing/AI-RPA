from execution.base_tool import BaseTool, ToolResult


class Terminate(BaseTool):
    """
    此工具用于终止当前交互。当遇到如下两种情况时，应立即调用此工具结束工作流程：
    1、用户请求已完全满足（所有任务已完成）；
    2、任务无法继续推进（遇到无法解决的障碍或资源限制）。
    """
    name: str = "terminate"
    description: str = """
    此工具用于终止当前交互。当遇到如下两种情况时，应立即调用此工具结束工作流程：
    1、用户请求已完全满足（用户给出的任务已完成），注意，如果你认为你已经完成了用户所给出的任务，不必询问用户是否需要进一步操作或帮助，直接调用此工具；
    2、任务无法继续推进（遇到无法解决的障碍或资源限制）。
    """
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "本交互的终止状态，success表示成功，failure表示失败。",
                "enum": ["success", "failure"],
            }
        },
        "required": ["status"],
    }

    def execute(self, status: str) -> str:
        """结束本次任务执行"""
        return ToolResult(output=f"本次任务结束。完成状态: {status}")
