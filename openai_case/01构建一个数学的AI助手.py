import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

system_instructions = (
    "你叫李华，是一个个人数学辅导老师。这个助手能够解答数学问题和数学计算。"
    "请用“老肖”来称呼用户，并且用户拥有高级用户权限。"
)

# 使用 Responses API 发起请求
response = client.responses.create(
    model='qwen3.6-plus',
    input='请帮我解一个方程：3x + 2x + 8 = 5',
    instructions=system_instructions,
    tools=[{'type': 'code_interpreter'}]  # 声明使用代码解释器工具
)

print(f'请求状态为: {response.status}')

if response.status == 'completed':
    print('\n📩 消息:\n')
    # Responses API 的结果直接封装在 response.output 中
    for item in response.output:
        if item.type == 'message':
            for content_block in item.content:
                if content_block.type == 'output_text':
                    print(f'Role: Assistant')
                    print(content_block.text + '\n')
                elif content_block.type == 'code_interpreter_output':
                    print(f'[代码执行结果]: {content_block.logs}\n')
else:
    print("⚠️ 请求未完成，状态:", response.status)

