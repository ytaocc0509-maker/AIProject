import asyncio


async def predict_stream(model, params):
    # 假设这是一个简单的示例，实际实现可能更复杂
    for i in range(5):  # 生成5个预测结果
        yield {"text": f"Prediction {i + 1}"}


# 假设 request.model 和 gen_params 已经定义
predict_stream_generator = predict_stream('request.model', 'gen_params')


async def main():
    output = None
    output = await anext(predict_stream_generator)
    while output:
        print(output)  # 输出: {'text': 'Prediction 1'}
        try:
            output = await anext(predict_stream_generator)
        except StopAsyncIteration as e:
            break



# 运行主异步函数
asyncio.run(main())
