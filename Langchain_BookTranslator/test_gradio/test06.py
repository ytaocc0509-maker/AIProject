import random
import time

import gradio as gr


def do_user(user_message, history):  # 把用户的问题消息，放到历史记录中
    # history.append((user_message, None))
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": ""})
    return '', history


def do_it(history):  # 定义一个回调函数，
    print(history)
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

    # 最后一条历史记录中，只有用户的提问消息，没有AI的的回答
    history[-1]["content"] = ''
    # 流式输出
    for char in resp:
        history[-1]["content"] += char  # 把最后一条聊天记录的  AI的回答 追加了一个字符
        time.sleep(0.1)
        yield history

css = """
#bgc {background-color: #7FFFD4}
.feedback textarea {font-size: 24px !important}
"""

# Blocks： 自定义各种组件联合的一个函数
with gr.Blocks(title='我的AI聊天机器人', css=css) as instance:  # 自定义
    gr.Label('我的AI聊天机器人', container=False)
    chatbot = gr.Chatbot(height=350, placeholder='<strong>AI机器人</strong><br> 你可以问任何问题')
    msg = gr.Textbox(placeholder='输入你的问题！', elem_classes='feedback', elem_id='bgc')
    clear = gr.ClearButton(value='清除聊天记录', components=[msg, chatbot])  # 清楚的按钮

    # 光标在文本输入框中，回车。 触发submit
    # 通过设置queue=False可以禁用队列，以便立即执行。
    #  在then里面：调用do_it函数，更新聊天历史，用机器人的回复替换之前创建的None消息，并逐字显示回复内容。
    msg.submit(do_user, [msg, chatbot], [msg, chatbot], queue=False).then(do_it, chatbot, chatbot)

# 启动服务
instance.queue()
instance.launch(server_name='0.0.0.0', server_port=8008)
