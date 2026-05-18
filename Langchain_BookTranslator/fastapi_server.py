import time

import uvicorn
from fastapi import FastAPI, Body, UploadFile, File
from starlette.responses import JSONResponse, FileResponse

from ai_model.openai_model import ChatQwenModel
from translator.book_translator import PDFTranslator
from utils.log_utils import log
from utils.project_config import ProjectConfig

# 创建fastapi的app
app = FastAPI()
TEMP_DIR = 'temp/'  # 存放用户上传的书籍文件 目录


@app.post('/translation', summary='调用AI翻译器', description='这是一个调用AI大模型的翻译器')
def translation(source_language: str = Body(description='源语言', default='English'),
                target_language: str = Body(description='目标语言', default='Chinese'),
                input_file: UploadFile = File(description='选择需要翻译的PDF文件')):  # 这就是一个API接口
    print('执行接口函数')
    try:
        if input_file.size > 0:
            # 保持到指定文件中
            # pdf_file_path = TEMP_DIR + int(time.time()) + ' ' + input_file.filename
            pdf_file_path = f'{TEMP_DIR}{int(time.time())}_{input_file.filename}'
            print(pdf_file_path)
            with open(pdf_file_path, 'wb') as f:
                f.write(input_file.file.read())
        output_file = translator.translate_book(file_path=pdf_file_path, source_language=source_language,
                                                target_language=target_language)

        return FileResponse(output_file, filename=input_file.filename)
    except Exception as e:
        log.exception(e)
        return JSONResponse({'detail': '翻译器报错了，请查看服务器日志!'}, status_code=500)


def init_translator():
    # 项目整体配置的初始化
    config = ProjectConfig()
    config.initialize()

    # 初始化大语言模型
    if config.model_type == 'ChatQwen':
        model = ChatQwenModel(config.model_name, config.api_key)
    else:
        print(config.model_type)
        model = ChatQwenModel(config.model_name, config.api_key)

    global translator
    translator = PDFTranslator(model)


if __name__ == '__main__':
    init_translator()
    # 先访问接口文档：
    # http://0.0.0.0:8000/docs
    # http://0.0.0.0:8000/redoc
    uvicorn.run(app, host='0.0.0.0', port=8000)