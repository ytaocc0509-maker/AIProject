import pandas as pd
from openai import OpenAI
import tiktoken

# 1、首先读取数据，然后预处理   df 是 pandas.DataFrame 类型
df = pd.read_csv('datas/fine_food_reviews_1k.csv', index_col=0)

# 返回一个 新的 DataFrame，包含指定的多列
df = df[['Time', 'ProductId', 'UserId', 'Score', 'Summary', 'Text']]

# 删除cvs中缺失的数据，NaN，NaT的数据
df = df.dropna()

# 合并'Summary', 'Text'   创建 DataFrame 列 combined
df['combined'] = "Title：" + df.Summary.str.strip() + "; Content：" + df.Text.str.strip()

# print(df.head(2))

# 2、生成Embedding之后的向量空间，并且保存

# 对文本进行处理的分词器

tokenizer_name = 'cl100k_base'
max_token = 8192  # 最大token
top_n = 10  # 最大评论数量
df = df.sort_values('Time')  # 按照时间排序
df.drop("Time", axis=1, inplace=True)  # 删掉Time

# 创建一个分词器
tokenizer = tiktoken.get_encoding(encoding_name=tokenizer_name)

# 控制输入数据的token数量
# 计算token数量
df['count_token'] = df.combined.apply(lambda x: len(tokenizer.encode(x)))

# token的数量不能超过官方的阈值: 超过了就不要。
# 在 pandas 中，当你用 布尔索引（Boolean Indexing） 过滤行时 df[condition]
# condition 只决定“保留哪些行”
# 不会影响“保留哪些列”
# 所有原始列都会被保留（只要该行被选中）
df = df[df.count_token <= max_token].tail(top_n)

# 初始化openai的客户端
client = OpenAI(  # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
    # 各地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key="sk-e7a344ef54e842dc8bfc289a7dce66bc",  # os.getenv("DASHSCOPE_API_KEY"),
    # 以下是北京地域base-url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def embedding_text(text, model="text-embedding-v4"):
    """
        通过Qwen的Embedding模型处理文本数据
    :param text: 需要处理的文本数据
    :param model:
    :return:

    :param text:
    :param model:
    :return:
    """
    resp = client.embeddings.create(input=text, model=model)# input=text 表示传入单个字符串。
    return resp.data[0].embedding # resp.data 是一个长度为 1 的列表取 [0] 得到第一个


df['embedding'] = df.combined.apply(embedding_text)  # 对 combined 列中的每一个文本，调用 embedding_text() 函数

df.to_csv('datas/embedding_output_1k.csv')

#print(df['embedding'][0])
