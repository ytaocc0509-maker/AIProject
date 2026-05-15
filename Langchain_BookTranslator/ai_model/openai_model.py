from langchain_qwq import ChatQwen

from ai_model.model import Model


class ChatQwenModel(Model):

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    def create_llm(self):
        """"
          初始化chatqwen大模型
        """
        return ChatQwen(model=self.model_name, api_key=self.api_key,
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", temperature=0.1)
