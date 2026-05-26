import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

# sqlalchemy 初始化MySQL数据库的连接
HOSTNAME = os.environ.get("MYSQL_HOST", "127.0.0.1")
PORT = os.environ.get("MYSQL_PORT", "3306")
DATABASE = os.environ.get("MYSQL_DATABASE", "world")
USERNAME = os.environ.get("MYSQL_USERNAME", "root")
PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
# mysqlclient驱动URL
MYSQL_URI = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)

# 创建模型
model = ChatOpenAI(
    temperature=1,
    model="glm-5.1",
    openai_api_key=os.getenv("Glm_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)
db = SQLDatabase.from_uri(MYSQL_URI)

create_sql = create_sql_query_chain(llm=model, db=db)

execute_sql = QuerySQLDataBaseTool(db=db)  # langchain内置的工具

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question. 用中文回答最终答案
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
)

answer_chain = answer_prompt | model | StrOutputParser()

# chain = create_sql | (lambda x: x.replace('```sql', '').replace('```', '')) | execute_sql

chain = RunnablePassthrough.assign(query=create_sql).assign(result=itemgetter('query') | execute_sql) | answer_chain

resp = chain.invoke({'question': '请问：一共有多少个国家？'})
# print("1:" + create_sql)
print(resp)

resp = chain.invoke({'question': '请问：哪个国家的人口最多？并且返回该国家的人口'})
# print("2:" +create_sql)
print(resp)
