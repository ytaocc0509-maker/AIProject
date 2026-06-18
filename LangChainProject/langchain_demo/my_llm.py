from langchain_openai import ChatOpenAI

from env_utils import LOCAL_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, OPENAI_API_KEY, OPENAI_BASE_URL

# openai的大模型
# llm = ChatOpenAI(
#     model='gpt-4o-mini',
#     temperature=0.8,
#     api_key=OPENAI_API_KEY,
#     base_url=OPENAI_BASE_URL,
# )

#  claude 的大模型调用
# llm = ChatOpenAI(
#     model='claude-3-7-sonnet-20250219',
#     temperature=0.8,
#     api_key=OPENAI_API_KEY,
#     base_url=OPENAI_BASE_URL,
# )

# 官方的deepseek
# llm = ChatOpenAI(
#     model='deepseek-reasoner',
#     # model='deepseek-chat',
#     temperature=0.8,
#     api_key=DEEPSEEK_API_KEY,
#     base_url=DEEPSEEK_BASE_URL,
#     model_kwargs={ "response_format": { "type": "json_object" } },
# )

# 本地私有化部署的大模型
llm = ChatOpenAI(
    model='qwen3-8b',
    temperature=0.8,
    api_key='xx',
    base_url=LOCAL_BASE_URL,
    extra_body={'chat_template_kwargs': {'enable_thinking': False}},
)

# llm = ChatOpenAI(
#     model='ds-qwen3-8b',
#     temperature=0.8,
#     api_key='',
#     base_url=LOCAL_BASE_URL
# )

multiModal_llm = ChatOpenAI(  # 多模态大模型
    model='qwen-omni-3b',
    api_key='xx',
    base_url=LOCAL_BASE_URL,
)