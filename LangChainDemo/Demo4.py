import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_qwq import ChatQwen
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import chat_agent_executor

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

search = TavilySearchResults(max_results=2)

# 让模型绑定工具
tools = [search]

# search_tool = StructuredTool.from_function(
#     func=search.invoke,
#     name="tavily_search",
#     description="使用Tavily搜索引擎搜索实时信息，如天气、新闻等"
# )

# model_with_tools = model.bind_tools([search_tool])

# resp = model_with_tools.invoke([HumanMessage(content='中国的首都是哪个城市？')])
# print(f'Model_Result_Content: {resp.content}')
# print(f'Tools_Result_Content: {resp.tool_calls}')

# resp2 = model_with_tools.invoke([HumanMessage(content='北京天气怎么样？')])
# print(f'Model_Result_Content: {resp2.content}')
# print(f'Tools_Result_Content: {resp2.tool_calls}')

agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)

resp = agent_executor.invoke({'messages': [HumanMessage(content='中国的首都是哪个城市？')]})
print(resp['messages'])

resp2 = agent_executor.invoke({'messages': [HumanMessage(content='北京天气怎么样？')]})
print(resp2['messages'])

print(resp2['messages'][2].content)