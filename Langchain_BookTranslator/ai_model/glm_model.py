import os

from ai_model.model import Model


class OpenAIModel(Model):

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    def create_llm(self):
        """"
          初始化chatqwen大模型
        """
        # model = ChatQwen(
        #     model="qwen-turbo",
        #     api_key=os.environ.get("DASHSCOPE_API_KEY"),
        #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        #     temperature=0.1
        # )
