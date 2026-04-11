import os

from ai_model.openai_model import OpenAIModel
from translator.book_translator import PDFTranslator
from utils.argument_utils import ArgumentUtils
from utils.loader_config import LoaderConfig

if __name__ == '__main__':
    # os.environ['http_proxy'] = '127.0.0.1:7000'
    # os.environ['https_proxy'] = '127.0.0.1:7000'

    # 启动命令中的参数解析和验证,并返回所有参数
    arg_utils = ArgumentUtils()
    args = arg_utils.parse_arg()

    # 读取配置文件（YAML）
    loader_config = LoaderConfig(args.config)
    config = loader_config.load_config()

    # 模型的名字： 先从命令行参数中获取，否则 从配置文件中获取
    model_name = args.openai_model if args.openai_model else config['OpenAIModel']['model']
    # api_key
    api_key = args.openai_api_key if args.openai_api_key else config['OpenAIModel']['api_key']

    # 初始化模型对象
    if args.model_type == 'OpenAIModel':
        model = OpenAIModel(model_name, api_key)
    else:
        # model = ChatGLMModel()
        pass


    # 初始化一个翻译器
    file_format: str = args.file_format if args.file_format else config['common']['file_format']
    # 得到入口文件的路径
    file_path: str = args.book if args.book else config['common']['book']
    if file_path[file_path.rindex('.'):] == '.pdf' or file_path[file_path.rindex('.'):] == '.PDF':
        translator = PDFTranslator(model)

    else:
        pass  # 如果需要翻译的书籍是doc或者docx。需要另外一个对象

    # 真正开启翻译书籍
    translator.translate_book(file_path, file_format)