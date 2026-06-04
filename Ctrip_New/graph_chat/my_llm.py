from langchain_openai import ChatOpenAI

from graph_chat.env_utils import ALIBABA_API_KEY, ALIBABA_BASE_URL

# 使用阿里云通义千问 Qwen3.7-Plus 模型
# 支持内置联网搜索功能，通过 extra_body 参数启用
llm = ChatOpenAI(
    model='qwen3.7-plus',
    temperature=0.6,
    streaming=True,
    api_key=ALIBABA_API_KEY,
    base_url=ALIBABA_BASE_URL,
    extra_body={
        "enable_search": True
    }
)