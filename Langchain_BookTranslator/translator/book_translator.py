from typing import Optional

from .file_writer import FileWriter
from .pdf_parser import parse_pdf
from ai_model.model import Model
from utils.log_utils import log
from .translator_chain import TranslatorChain


class PDFTranslator:
    """
    翻译一个pdf文件的书籍
    """

    def __init__(self, model: Model):
        self.book = None
        self.chain = TranslatorChain(model) # 真正负责翻译文本内容
        self.writer = FileWriter(self.book)

    def translate_book(self, file_path: str, source_language: str, target_language: str, out_file_format: str = 'PDF',
                       out_file_path: str = None, pages: Optional[int] = None):
        """
        翻译指定的一本书
        :param file_path:
        :param source_language:  书籍中的原始语言
        :param target_language:
        :param out_file_format:
        :param out_file_path:
        :param pages:  需要翻译多少页，如果不传，则翻译整本书
        :return:
        """
        self.book = parse_pdf(file_path, pages)  # 解析文件得到一个book对象

        self.writer.book = self.book
        for page_index, page in enumerate(self.book.pages):  # 遍历所有的页
            for content_index, content in enumerate(page.contents):
                # 开始翻译每一个内容, 使用langchain
                translation_text, status = self.chain.run(content, source_language, target_language)
                log.debug(f'翻译之后的内容是: \n {translation_text}')
                # 把翻译之后的文本和状态设置到content对象中 （保存）
                self.book.pages[page_index].contents[content_index].set_translation(translation_text, status)

        # 把翻译之后的所有数据，写入文件
        # Writer
        output_path = self.writer.save_book(out_file_path, out_file_format)
        return output_path
