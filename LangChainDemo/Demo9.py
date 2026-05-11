import os
from dotenv import load_dotenv
from langchain_qwq import ChatQwen
from langchain_experimental.synthetic_data import create_data_generation_chain

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_TRACING_V2"] = os.environ.get("LANGCHAIN_TRACING_V2", "false")
os.environ["TAVILY_API_KEY"] = os.environ.get("TAVILY_API_KEY")

# 创建模型
model = ChatQwen(
    model="qwen-turbo",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.8
)

# 创建链
chain = create_data_generation_chain(model)

# 生成数据
result = chain(  # 给于一些关键词， 随机生成一句话
    {
        "fields": ['蓝色', '黄色'],
        "preferences": {"language":"中文","style":"让它像诗歌一样"}
    }
)
print(result)