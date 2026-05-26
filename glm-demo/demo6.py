import os

from dotenv import load_dotenv
from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
# from langchain_community.tools import QuerySQLDataBaseTool
from langchain_openai import ChatOpenAI
from langchain_qwq import ChatQwen
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
    model="glm-5",
    openai_api_key=os.getenv("Glm_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)
# model = ChatQwen(
#     model="qwen-turbo",
#     api_key=os.environ.get("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     temperature=0
# )

db = SQLDatabase.from_uri(MYSQL_URI)

# print(db.get_usable_table_names())
# print(db.run('select * from city limit 10;'))

# 清理SQL前缀的函数（移除 "SQLQuery: " 等前缀）
def clean_sql_query(query):
    if isinstance(query, str):
        # 移除常见的前缀
        prefixes = ["SQLQuery:", "SQL:", "Query:", "sql:", "query:", "```sql", "```"]
        cleaned = query.strip()
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        # 移除末尾的反引号
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return cleaned
    return query


chian = create_sql_query_chain(llm=model, db=db)
# chian.get_prompts()[0].pretty_print()
resp = chian.invoke({'question': '请问：city表中有多少条数据？'})
# print(resp)

print(db.run(resp))