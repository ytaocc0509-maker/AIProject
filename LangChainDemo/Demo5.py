import os
import bs4
from dotenv import load_dotenv
from langchain_qwq import ChatQwen
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_TRACING_V2"] = os.environ.get("LANGCHAIN_TRACING_V2", "false")
os.environ["TAVILY_API_KEY"] = os.environ.get("TAVILY_API_KEY")

model = ChatQwen(
    model="qwen-turbo",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.1
)

# 1、加载数据: 一篇博客内容数据
loader = WebBaseLoader(
    web_paths=['https://lilianweng.github.io/posts/2023-06-23-agent/'],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=('post-header', 'post-title', 'post-content'))
    )
)

# docs = loader.load()

# print(len(docs))
# print(docs)

# 2、大文本的切割
text = "hello world, how about you? thanks, I am fine.  the machine learning class. So what I wanna do today is just spend a little time going over the logistics of the class, and then we'll start to talk a bit about machine learning"

splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=4)

res = splitter.split_text(text)

for s in res:
    print(s,end="\n")
