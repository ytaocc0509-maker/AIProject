from ai_model.model import Model
from book.content import Content, ContentType
from utils.log_utils import log


class TranslatorChain:
    """
    负责调用langchain来完成文本的翻译， chain对象，要不要是单例的？
    """
    _instance = None

    def __init__(self, model: Model):
        # 初始化得到一个链
        self.langchain = model.make_prompt() | model.create_llm()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TranslatorChain, cls).__new__(cls)
        return cls._instance

    def run(self, content: Content, source_language: str, target_language: str):
        """
        翻译具体的文本，返回翻译之后的文本内容和翻译成功的状态
        :param content:
        :param source_language:
        :param target_language:
        :return: translation_text, status
        """
        text = ''
        result = ''
        try:
            # 提示模板中的三个变量
            if content.content_type == ContentType.TEXT:
                text = f'请按照要求翻译以下的内容: {content.original}'
            elif content.content_type == ContentType.TABLE:
                log.info("+++++++++++++")
                text = f'请将以下表格中的所有文字从{source_language}翻译成{target_language}，保持表格的行数和列数不变，只翻译文字内容，数字保持不变。每行的元素之间用英文逗号分隔，每行之间用换行符分隔：\n {content.get_original_to_string()}'

            result = self.langchain.invoke({
                'source_language': source_language,
                'target_language': target_language,
                'text': text
            })
            log.info(result.content)
        except Exception as e:
            log.exception(e)
            return result, False  # 报错的返回
        return result.content, True