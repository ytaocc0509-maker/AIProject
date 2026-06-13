from zhipuai import ZhipuAI

from env_utils import ZHIPU_API_KEY

# 初始化 ZhipuAI 客户端
client = ZhipuAI(api_key=ZHIPU_API_KEY)

# 定义工具参数
tools = [{
    "type": "web_search",
    "web_search": {
        "enable": True,
        "search_engine": "search_pro_sogou",  # 选择搜索引擎类型，默认为基础版
        "search_result": True,
        "search_prompt": "你是一名财经分析师，请用简洁的语言总结网络搜索中：{{search_result}}中的关键信息，按重要性排序并标注来源日期。当前日期是2025年 4 月 11 日"
    }
}]

# 定义用户消息
messages = [{
    "role": "user",
    "content": "2025年4月重要财经事件 政策变化 市场数据"
}]

# 调用 API 获取响应
response = client.chat.completions.create(
    model="glm-4-air-250414",  # 模型编码
    messages=messages,  # 用户消息
    tools=tools  # 工具参数
)

# 打印响应结果
# print(response.choices)

for choice in response.choices:
    print(choice.message.content)
    print('---' * 20)