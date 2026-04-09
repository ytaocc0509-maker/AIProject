import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 第一轮的聊天

message = [
    {'role': 'system', 'content': '你是一个无所不能的体育专家。'},
    {
        'role': 'user',
        'content': '你好啊！'
    }
]
result = client.chat.completions.create(
    model='qwen-plus',
    messages=message,
    max_tokens=100
)

# print(result)
out_result = result.choices[0].message
print(out_result.content)

message.append({'role': out_result.role, 'content': out_result.content})

# 第二轮的聊天

new_chat = {
    'role': 'user',
    'content': '1、2024年的奥运会在哪个国家举行。2、告诉我这个国家的首都是什么？'
}

message.append(new_chat)

result = client.chat.completions.create(
    model='qwen-plus',
    messages=message,
    max_tokens=100
)
out_result = result.choices[0].message
print(out_result.content)

message.append({'role': out_result.role, 'content': out_result.content})

# 开启第三轮对话：
new_chat = {
    'role': 'user',
    'content': '请问这个国家历史上一共获得过多少金牌？'
}

message.append(new_chat)

result = client.chat.completions.create(
    model='qwen-plus',
    messages=message,
    max_tokens=100
)
out_result = result.choices[0].message
print(out_result.content)

