import os
from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_qwq import ChatQwen

from langchain.agents import create_agent

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_TRACING_V2"] = os.environ.get("LANGCHAIN_TRACING_V2", "false")
os.environ["TAVILY_API_KEY"] = os.environ.get("TAVILY_API_KEY")

# 创建模型
model = ChatQwen(
    model="qwen-turbo",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.1
)

# sqlalchemy 初始化MySQL数据库的连接
HOSTNAME = os.environ.get("MYSQL_HOST", "127.0.0.1")
PORT = os.environ.get("MYSQL_PORT", "3306")
DATABASE = os.environ.get("MYSQL_DATABASE", "world")
USERNAME = os.environ.get("MYSQL_USERNAME", "root")
PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
# mysqlclient驱动URL
MYSQL_URI = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)

db = SQLDatabase.from_uri(MYSQL_URI)

# 创建工具
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# 使用agent完整整个数据库的整合
system_prompt = """
您是一个被设计用来与SQL数据库交互的代理。
给定一个输入问题，创建一个语法正确的SQL语句并执行，然后查看查询结果并返回答案。
除非用户指定了他们想要获得的示例的具体数量，否则始终将SQL查询限制为最多10个结果。
你可以按相关列对结果进行排序，以返回MySQL数据库中最匹配的数据。
您可以使用与数据库交互的工具。在执行查询之前，你必须仔细检查。如果在执行查询时出现错误，请重写查询SQL并重试。
不要对数据库做任何DML语句(插入，更新，删除，删除等)。

首先，你应该查看数据库中的表，看看可以查询什么。
不要跳过这一步。
然后查询最相关的表的模式。
"""
system_message = SystemMessage(content=system_prompt)

# 创建代理
agent_executor = create_agent(model, tools, system_prompt=system_prompt)

# resp = agent_executor.invoke({'messages': [HumanMessage(content='city表中有多少个城市？')]})
# resp = agent_executor.invoke({'messages': [HumanMessage(content='哪个国家拥有最多的城市?')]})
resp = agent_executor.invoke({'messages': [HumanMessage(content='哪个国家拥有最少少人口的城市？')]})

result = resp['messages']
print(result)
print(len(result))
# 最后一个才是真正的答案
print(result[len(result) - 1])
