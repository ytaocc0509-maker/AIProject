from fastapi import FastAPI
import uvicorn

# 创建fastapi的app
app = FastAPI()


@app.get('/translation')
def test():  # 这就是一个API接口
    print('执行接口函数')
    return 'Hello'


if __name__ == '__main__':
    # 先访问接口文档：
    # http://0.0.0.0:8000/docs
    # http://0.0.0.0:8000/redoc
    uvicorn.run(app, host='0.0.0.0', port=8000)
