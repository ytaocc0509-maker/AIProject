import os

from dotenv import load_dotenv
from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
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

chian = create_sql_query_chain(llm=model, db=db)

resp = chian.invoke({'question': '请问：city表中有多少条数据？'})
print(resp)

# print(db.run(resp))