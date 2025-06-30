from prompt import SYSTEM_PROMPT, NEXT_STEP_PROMPT
from openai import OpenAI
from dotenv import load_dotenv
import json
from function import load_tools

load_dotenv()  # 从 .env 加载环境变量

class Client:
    def __init__(self, name):
        # 初始化会话和客户端对象
        self.openai = OpenAI(api_key="sk-49afb3447ba144e99f2b02b9b1e6d134", base_url="https://api.deepseek.com")
        self.model = 'deepseek-chat'
        self.stop_flag = False
        self.need_human = True
        self.tools, self.classes = load_tools()
        self.system_prompt = SYSTEM_PROMPT.format(name=name, directory="/Users/mengheqing/PycharmProjects/AI-RPA/")
        self.next_step_prompt = NEXT_STEP_PROMPT
        # self.messages = []
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": self.next_step_prompt}
        ]

    def chat_loop(self):
        """运行交互式聊天循环"""
        print("\nMCP 客户端已启动！")
        print("输入你的查询或输入 'quit' 退出。")

        while True:
            if 1:
                if self.need_human:
                    query = input("\n请输入你的查询: ").strip()
                    if query.lower() == 'quit':
                        break
                    self.need_human = False
                else:
                    query = "做的很好，继续。"

                if query.lower() == 'quit':
                    break

                response = self.process_query(query)
                print("\n" + response)

            if self.stop_flag:
                break

            # except Exception as e:
            #     print(f"\n错误: {str(e)}")

    def cleanup(self):
        """清理资源"""
        return

    def connect_to_server(self):
        """连接到 MCP 服务器"""
        print("\n已连接到服务器，工具包括：", [tool['function']['name'] for tool in self.tools])

    def process_query(self, query: str) -> str:
        """使用可用的工具处理查询"""
        final_text = []
        self.messages.append(
            {
                "role": "user",
                "content": query
            }
        )
        response = self.openai.chat.completions.create(model=self.model, messages=self.messages, stream=False, tools=self.tools, tool_choice="auto")
        response = response.choices[0]
        if response.finish_reason == 'stop':
            # self.messages.append(response.message)
            self.messages.append(
                {
                    "role": response.message.role,
                    "content": response.message.content
                }
            )
            final_text.append(response.message.content)
        elif response.finish_reason == 'tool_calls':
            self.messages.append(
                {
                    "role": response.message.role,
                    "content": response.message.content,
                    "tool_calls": response.message.tool_calls
                }
            )
            for target_tool in response.message.tool_calls:

                tool_call_id = target_tool.id
                tool_name = target_tool.function.name
                arguments = json.loads(target_tool.function.arguments)

                # print('调用工具：', tool_name)

                if tool_name == 'terminate':
                    self.stop_flag = True

                if tool_name == 'human_help':
                    self.need_human = True

                # 获取目标类
                target_class = self.classes[tool_name]
                # 创建类的实例
                instance = target_class()
                # 获取目标方法
                method = getattr(instance, 'execute')
                # 使用字典解包传递参数调用方法
                result = method(**arguments)
                result_set = result.model_fields_set

                for r in result_set:
                    if r == 'base64_image':
                        result = result.base64_image
                    elif r == 'output':
                        result = result.output
                    elif r == 'error':
                        result = result.error
                    elif r == 'system':
                        result = result.system
                    else:
                        result = result.output

                # print('调用结果：', result)
                self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})

            response = self.openai.chat.completions.create(model=self.model, messages=self.messages, stream=False, tools=self.tools)
            final_text.append(response.choices[0].message.content)

        return "\n".join(final_text)


def main(name):
    client = Client(name)
    # try:
    #     client.connect_to_server()
    #     client.chat_loop()
    # finally:
    #     client.cleanup()
    client.connect_to_server()
    client.chat_loop()


if __name__ == "__main__":
    import sys

    name = "黄建亮"
    main(name)