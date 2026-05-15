from book.page import Page


class Book:
    """
    代表你需要翻译的一本书
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.pages: [Page] = []  # 这本所有的内容页

    def add_page(self, page: Page):
        self.pages.append(page)
