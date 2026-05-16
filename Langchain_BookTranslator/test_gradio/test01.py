import gradio as gr


def calculator(num1, operation, num2):
    if operation == '加':
        return num1 + num2
    elif operation == '减':
        return num1 - num2
    elif operation == '乘':
        return num1 * num2
    elif operation == '除':
        if num2 == 0:
            raise gr.Error('0不能作为除数！')
        return num1 / num2


instance = gr.Interface(  # 构建一个UI界面
    fn=calculator,
    inputs=[
        'number',
        gr.Radio(choices=['加', '减', '乘', '除'], label='计算法则'),
        'number'
    ],
    outputs='number'
)

# 启动服务
instance.launch(server_name='0.0.0.0', server_port=8008, auth=('admin', '123123'))