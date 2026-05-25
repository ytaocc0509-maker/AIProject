from zai import ZhipuAiClient
import os

# 从环境变量读取 API Key
client = ZhipuAiClient(api_key=os.getenv("Glm_API_KEY"))

response = client.chat.completions.create(
    model="glm-5-turbo",
    messages=[
        {"role": "user", "content": "作为一名营销专家，请为我的产品创作一个吸引人的口号"},
        {"role": "assistant", "content": "当然，要创作一个吸引人的口号，请告诉我一些关于您产品的信息"},
        {"role": "user", "content": "智谱开放平台"}
    ],
    thinking={
        "type": "disabled",    # 启用深度思考模式
    },
    max_tokens=1000,          # 最大输出 tokens
    temperature=1.0           # 控制输出的随机性
)

# 获取完整回复
print(response.choices[0].message)
