import pandas as pd
import numpy as np
import ast
from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API配置
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")  # 从环境变量读取
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v4"

# 初始化openai的客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
df = pd.read_csv('datas/embedding_output_1k.csv')

# 把str转化成矩阵
df['embedding_vec'] = df['embedding'].apply(ast.literal_eval)


def embedding_text(text, model=EMBEDDING_MODEL):
    """
    通过OpenAI的Embedding模型处理文本数据
    :param text: 需要处理的文本数据
    :param model:
    :return:
    """
    resp = client.embeddings.create(input=text, model=model)
    return resp.data[0].embedding


# 在向量空间中，语义相似的词或者文本单位。距离是靠近
def cosine_distance(a, b):
    """
    计算两个向量之间的余弦距离
    :param a:
    :param b:
    :return:
    """
    # 得到这两个向量之间的夹角余弦值，如果余弦相似度接近于 1，则表示这两个向量非常相似；接近于 -1 表示它们方向相反；
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_by_word(df, work_key, n_result=3, print_flag=True):
    """
    根据指定的关键词(句子)，去向量空间中进行相似搜索
    :param df: 已经Embedding之后的向量空间
    :param work_key:
    :param n_result: 返回结果中的数量
    :param print_flag: 是否打印
    :return:
    """
    # 把关键词向量化
    word_embedding = embedding_text(work_key)

    # 计算相似度
    df['similarity'] = df.embedding_vec.apply(lambda x: cosine_distance(x, word_embedding))

    res = (
        df.sort_values('similarity', ascending=False)
        .head(n_result)
        .combined.str.replace('Title：', "")
        .str.replace('; Content：', ';')
    )

    if print_flag:
        for r in res:
            print(r)
            print()
    return res


if __name__ == '__main__':
    search_by_word(df, 'delicious beans', 3)
    print('=' * 50)
    search_by_word(df, 'food', 3)
