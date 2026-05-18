import  argparse

class ArgumentUtils:
    """
    命令行参数解析对象
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='书籍自动翻译器')
        self.parser.add_argument('--config', type=str, default='config.yaml', help='项目的整体配置文件')
        self.parser.add_argument('--model_type', type=str, default='ChatQwen', choices=['GLMModel', 'ChatQwen'],
                                 help='选择OpenAI还是GLM的模型')
        self.parser.add_argument('--model_name', type=str, help='大预言模型的名字')
        self.parser.add_argument('--input_file', type=str, help='需要翻译的书籍所属的文件路径')
        self.parser.add_argument('--file_format', type=str, help='翻译之后生成的文件格式')
        self.parser.add_argument('--source_language', type=str, help='书记的原始语言')
        self.parser.add_argument('--target_language', type=str, help='翻译之后的语言')

    def parse_arg(self):
        """
        解析和验证命令中的参数
        :return:
        """
        # 解析参数
        args = self.parser.parse_args()
        # 参数的验证，可以有更多
        # if args.model_type == 'OpenAIModel' and not args.openai_model and not args.openai_api_key:
        #     self.parser.error('当选择OpenAI后，--openai_model和--openai_api_key参数，必须要传！')
        return args
