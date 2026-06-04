import os

from dotenv import load_dotenv

load_dotenv(override=True)

# 阿里云通义千问 API 配置
ALIBABA_API_KEY = os.getenv('ALIBABA_API_KEY')
ALIBABA_BASE_URL = os.getenv('ALIBABA_BASE_URL')