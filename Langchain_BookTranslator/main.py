from ai_model.openai_model import ChatQwenModel
from translator.book_translator import PDFTranslator
from utils.project_config import ProjectConfig

if __name__ == '__main__':
    # 项目整体配置的初始化
    config = ProjectConfig()
    config.initialize()

    # 初始化大语言模型
    if config.model_type == 'ChatQwen':
        model = ChatQwenModel(config.model_name, config.api_key)
    else:
        model = ChatQwenModel(config.model_name, config.api_key)



    translator = PDFTranslator(model)
    translator.translate_book(file_path=config.input_file, source_language=config.source_language,
                              target_language=config.target_language, out_file_format=config.file_format)