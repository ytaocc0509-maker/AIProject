import random
import time

import gradio as gr


def do_it(message, history):   # 定义一个回调函数，
    responses = [
        "谢谢您的留言！",
        "非常有趣！",
        "我不确定该如何回答。",
        "请问还有其他问题吗？",
        "我会尽快回复您的。",
        "很高兴能与您交流！",
    ]
    # 生成一个答案，随机
    resp = random.choice(responses)
    # 流式输出
    res = ''
    for char in resp:
        res += char
        time.sleep(0.1)
        yield res


instance = gr.ChatInterface(  # 构建一个UI界面
    fn=do_it,
    chatbot=gr.Chatbot(height=350,placeholder='<strong>AI机器人</strong><br>'),
    textbox=gr.Textbox(placeholder='输入你的问题。',container=False,scale=8),
    title='我的聊天AI机器人',
    description='AI助手',
    examples=['Hello',"今天天气怎么样？"],
    cache_examples=True,
    # undo_btn=None,
    # clear_btn='清除记录'

)

# 启动服务
instance.launch(server_name='0.0.0.0', server_port=8008)
