import os
from dotenv import load_dotenv
from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_qwq import ChatQwen
from operator import itemgetter

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

# 测试连接是否成功
# print(db.get_usable_table_names())
# print(db.run('select * from city limit 10;'))

test_chain = create_sql_query_chain(model, db)
# resp = test_chain.invoke({'question': '请问：city表中有多少条数据？'})
# print(resp)
answer_prompt = PromptTemplate.from_template(
    """给定以下用户问题、SQL语句和SQL执行后的结果，回答用户问题。
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    回答: """
)

# 创建一个执行sql语句的工具
execute_sql_tool = QuerySQLDataBaseTool(db=db)

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

# 1、生成SQL，2、清理SQL，3、执行SQL
# 2、模板
chain = (RunnablePassthrough.assign(query=test_chain)
         .assign(cleaned_query=itemgetter('query') | RunnableLambda(clean_sql_query))
         .assign(result=itemgetter('cleaned_query') | execute_sql_tool)
         | answer_prompt
         | model
         | StrOutputParser()
         )

rep = chain.invoke(input={'question': '请问：city表中有多少条数据'})
print(rep)