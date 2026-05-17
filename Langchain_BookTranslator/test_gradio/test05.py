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
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": resp})
    time.sleep(1)
    return '', history     # 自动清除文本输入框中的内容


# Blocks： 自定义各种组件联合的一个函数

with gr.Blocks(title='我的AI聊天机器人') as instance:  # 自定义
    chatbot = gr.Chatbot(height=350, placeholder='<strong>AI机器人</strong><br> 你可以问任何问题')
    msg = gr.Textbox(placeholder='输入你的问题！', container=False)
    clear = gr.ClearButton(value='清除聊天记录', components=[msg, chatbot])  # 清楚的按钮

    msg.submit(do_it, [msg, chatbot], [msg, chatbot])  # 光标在文本输入框中，回车。 触发submit




# 启动服务
instance.launch(server_name='0.0.0.0', server_port=8008)
