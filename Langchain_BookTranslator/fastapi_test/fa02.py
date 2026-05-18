from fastapi import FastAPI, Body, UploadFile, File
import uvicorn
import os
from starlette.responses import FileResponse

# 创建fastapi的app
app = FastAPI()


@app.post('/translation', summary='调用AI翻译器', description='这是一个调用AI大模型的翻译器')
def test(source_language: str = Body(description='源语言', default='English'), target_language: str = Body(description='目标语言', default='Chinese'), input_file: UploadFile = File(description='选择需要翻译的PDF文件')):  # 这就是一个API接口
    print('执行接口函数')
    print(source_language)
    print(target_language)
    print(input_file.filename)

    print(input_file.size)
    # 1、相对路径
    output_file_path = '../test/tset.txt'
    # 2、绝对路径
    output_file_path = os.path.dirname(os.getcwd()) + '/test/test.txt'
    print(output_file_path)
    return FileResponse(output_file_path, filename='test2.txt')

if __name__ == '__main__':
    # 先访问接口文档：
    # http://0.0.0.0:8000/docs
    # http://0.0.0.0:8000/redoc
    uvicorn.run(app, host='0.0.0.0', port=8000)
