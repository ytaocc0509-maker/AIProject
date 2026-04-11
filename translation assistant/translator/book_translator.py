from typing import Optional

from .file_writer import FileWriter
from .pdf_parser import parse_pdf
from ai_model.model import Model
from utils.log_utils import log


class PDFTranslator:
    """
    翻译一个pdf文件的书籍
    """

    def __init__(self, model: Model):
        self.book = None
        self.model = model
        self.writer = FileWriter(self.book)

    def translate_book(self, file_path: str, out_file_format: str = 'PDF', target_language: str = '中文',
                       out_file_path: str = None, pages: Optional[int] = None):
        """
        翻译一本书
        :param file_path:
        :param out_file_format:
        :param target_language:
        :param out_file_path:
        :param pages:
        :return:
        """
        self.book = parse_pdf(file_path, pages)  # 解析文件得到一个book对象

        self.writer.book = self.book
        for page_index, page in enumerate(self.book.pages):  # 遍历所有的页
            for content_index, content in enumerate(page.contents):
                # 开始翻译每一个内容
                # 1、得到翻译的提示文本
                prompt = self.model.make_prompt(content, target_language)
                log.debug('大语言模型的提示信息:' + prompt)
                translation_text, status = self.model.request_model(prompt)

                log.debug(f'翻译之后的内容是: \n {translation_text}')
                # 把翻译之后的文本和状态设置到content对象中 （保存）
                self.book.pages[page_index].contents[content_index].set_translation(translation_text, status)

        # 把翻译之后的所有数据，写入文件
        # Writer
        self.writer.save_book(out_file_path, out_file_format)
