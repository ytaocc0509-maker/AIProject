from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_qwq import ChatQwen
class Model:
    """
    AI的模型对象
    """
    def create_llm(self):
        print("初始化大预言模型的对象")


    def make_prompt(self):
        """
        创建提示模板
        """
        system_template = """你是一个翻译专家，精通各种人类语言。\n
        输入的是：{source_language} 语言，翻译之后的语言为：{target_language}"""
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template('{text}')  # 用户真正翻译的提示是动态的

        return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


        # if content.content_type == ContentType.TEXT:
        #     return f'请翻译成{target_language}: {content.original}'
        # if content.content_type == ContentType.TABLE:
        #     return f'请翻译成{target_language}，每个元素之间用逗号隔开，以非MarkDown的表格形式返回：\n {content.get_original_to_string()}'