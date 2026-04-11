from book.content import ContentType


class Model:
    """
    AI的模型对象
    """

    def request_model(self, prompt):
        """
        请求模型的API接口
        :param prompt:
        :return:
        """
        print('发送请求，调用openai或者glm模型')

    def make_prompt(self, content, target_language):
        """
        创建发给 大语言模型的提示文本
        :param content:
        :param target_language:
        :return:
        """
        if content.content_type == ContentType.TEXT:
            return f'请翻译成{target_language}: {content.original}'
        if content.content_type == ContentType.TABLE:
            return f'请翻译成{target_language}，每个元素之间用逗号隔开，以非MarkDown的表格形式返回：\n {content.get_original_to_string()}'