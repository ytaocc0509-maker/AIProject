import os

from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_openai import ChatOpenAI

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
    model="glm-5",
    openai_api_key=os.getenv("Glm_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)
db = SQLDatabase.from_uri(MYSQL_URI)

create_sql = create_sql_query_chain(llm=model, db=db)

execute_sql = QuerySQLDataBaseTool(db=db)  # langchain内置的工具

chain = create_sql | (lambda x: x.replace('```sql', '').replace('```', '')) | execute_sql

resp = chain.invoke({'question': '请问：一共有多少个员工？'})

print(resp)

# resp = chian.invoke({'question': '请问：一共有多少个员工？'})
# print('大语言模型生成的SQL：' + resp)
# sql = resp.replace('```sql', '').replace('```', '')
# print('提取之后的SQL：' + sql)
#
# print(db.run(sql))