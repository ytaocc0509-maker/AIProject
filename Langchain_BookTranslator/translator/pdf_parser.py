from typing import Optional

import pdfplumber
from utils.log_utils import log
from book.book import Book
from book.content import Content, ContentType, TableContent
from book.page import Page
from utils.exceptions import PageOutOfRangeException


def parse_pdf(pdf_file_path: str, pages: Optional[int] = None) -> Book:
    """
    解析pdf文件的函数，返回解析之后的文本对象
    :param pdf_file_path: pdf文件路径
    :param pages: 可选的， 需要翻译前n页。默认就是整个pdf文件
    :return: 返回一个Book对象
    """
    book = Book(pdf_file_path)  # 一个pdf对应一本书，就是一个book对象

    with pdfplumber.open(pdf_file_path) as pdf:

        # pages不能超过pdf文件中的总页数
        if pages and pages > len(pdf.pages):  # 页数超过范围
            raise PageOutOfRangeException(len(pdf.pages), pages)

        if pages is None:  # 如果pages没有传，则翻译整本书
            pages_arr = pdf.pages
        else:
            pages_arr = pdf.pages[:pages]  # 通过切片截取前pages个页面

        for pdf_page in pages_arr:  # 遍历每一页
            page = Page()  # 每一页就是一个page对象

            # 从pdf的page中提取文本内容
            raw_text = pdf_page.extract_text()
            tables = pdf_page.extract_tables()
            # 出现重复的文本提取

            # 从raw_text中删除，表格中已经存在的文本
            for table in tables:
                for row in table:
                    for cell in row:
                        raw_text = raw_text.replace(cell, '', 1)

            # 处理文本内容
            if raw_text:
                # 数据清洗： 删除空行，和首尾空白字符
                lines = raw_text.splitlines()
                cleaned_lines = [line.strip() for line in lines if line.strip()]
                cleaned_text = '\n'.join(cleaned_lines)

                # 文本内容对应一个Content对象
                text_content = Content(content_type=ContentType.TEXT, original=cleaned_text)
                page.add_content(text_content)  # 把文本内容添加到 page中去
                log.debug(f'[pdf解析之后的文本内容]: \n{cleaned_text}')

            # 处理所有表格数据
            if tables:
                tables_content = TableContent(content_type=ContentType.TABLE, original=tables)
                page.add_content(tables_content)  # 把表格内容添加到 page中去
                log.debug(f'[pdf解析之后的表格内容]: \n{tables}')

            book.add_page(page)  # 把每一个page对象添加到book中
    return book


# python-docx