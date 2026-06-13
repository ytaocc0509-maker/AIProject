from langchain_openai import ChatOpenAI
from zhipuai import ZhipuAI

from env_utils import ZHIPU_API_KEY

zhipuai_client = ZhipuAI(api_key=ZHIPU_API_KEY)  # 填写您自己的APIKey


llm = ChatOpenAI(  # zhipuai的
    temperature=0,
    model='glm-4-air-250414',
    api_key=ZHIPU_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/")

