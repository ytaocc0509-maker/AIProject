import time

import gradio as gr


def do_it(input_word: str, progress=gr.Progress()):   # 定义一个回调函数，并初始化一个进度条
    res = ''
    progress(0, desc='开始...')

    # 进度条滚动，
    for letter in progress.tqdm(input_word, desc='运行中...'):
        time.sleep(0.25)
        res = res + letter
    return res


instance = gr.Interface(  # 构建一个UI界面
    fn=do_it,
    inputs=[
       gr.Text(label='请输入任何文本')
    ],
    outputs=gr.Text(label='输出的结果：')
)

# 启动服务
instance.launch(server_name='0.0.0.0', server_port=8008)
