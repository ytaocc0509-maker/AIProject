from modelscope import snapshot_download
model_dir = snapshot_download('ZhipuAI/glm-4-9b-chat', cache_dir='/root/autodl-tmp/models', revision='master')