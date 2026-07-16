import os
import subprocess

# text-model THUDM/glm-4-9b-chat
MODEL_PATH = os.environ.get('MODEL_PATH', '/root/autodl-tmp/models/ZhipuAI/glm-4-9b-chat')

# vision-model THUDM/glm-4v-9b
# MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/glm-4v-9b')


if '4v' in MODEL_PATH.lower():
    subprocess.run(["python", "glm4v_server.py", MODEL_PATH])
else:
    # 开启一个子进程，运行python命令
    subprocess.run(["python", "glm_server.py", MODEL_PATH])