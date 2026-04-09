import pandas as pd
from openai import OpenAI
import tiktoken
import os
import time
from tqdm import tqdm
import logging
from typing import List, Dict, Optional
import numpy as np
from dotenv import load_dotenv

# ==================== 配置管理 ====================
class Config:
    """配置参数集中管理"""
    # 文件路径
    INPUT_FILE = 'datas/fine_food_reviews_1k.csv'
    OUTPUT_FILE = 'datas/embedding_output_1k.csv'

    # 列配置
    COLUMNS = ['Time', 'ProductId', 'UserId', 'Score', 'Summary', 'Text']
    COMBINED_TEMPLATE = "Title：{summary}; Content：{content}"

    # 分词器配置
    TOKENIZER_NAME = 'cl100k_base'
    MAX_TOKENS = 8192

    # 处理配置
    TOP_N = 1000
    BATCH_SIZE = 10  # 批量处理大小

    # 加载环境变量
    load_dotenv()

    # API配置
    API_KEY = os.getenv("DASHSCOPE_API_KEY", "")  # 从环境变量读取
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBEDDING_MODEL = "text-embedding-v4"

    # 日志配置
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'


# ==================== 日志配置 ====================
logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


# ==================== 数据加载与预处理 ====================
def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    加载CSV数据并进行预处理

    Args:
        filepath: CSV文件路径

    Returns:
        预处理后的DataFrame
    """
    logger.info(f"正在加载数据: {filepath}")

    try:
        # 读取数据
        df = pd.read_csv(filepath, index_col=0)

        # 选择需要的列
        df = df[Config.COLUMNS]

        # 删除缺失值
        initial_count = len(df)
        df = df.dropna()
        logger.info(f"删除 {initial_count - len(df)} 行缺失数据，剩余 {len(df)} 行")

        # 合并Summary和Text
        df['combined'] = df.apply(
            lambda row: Config.COMBINED_TEMPLATE.format(
                summary=row['Summary'].strip(),
                content=row['Text'].strip()
            ),
            axis=1
        )

        # 删除原始列以节省内存
        df.drop(['Summary', 'Text'], axis=1, inplace=True)

        return df

    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        raise


# ==================== Token计算与过滤 ====================
def filter_by_token_limit(df: pd.DataFrame, max_tokens: int = Config.MAX_TOKENS) -> pd.DataFrame:
    """
    计算token数量并过滤超出限制的文本

    Args:
        df: DataFrame
        max_tokens: 最大token限制

    Returns:
        过滤后的DataFrame
    """
    logger.info("开始计算token数量...")

    # 创建分词器
    tokenizer = tiktoken.get_encoding(encoding_name=Config.TOKENIZER_NAME)

    # 计算token数量
    df['count_token'] = df['combined'].apply(lambda x: len(tokenizer.encode(x)))

    # 过滤超出限制的数据
    before_filter = len(df)
    df = df[df['count_token'] <= max_tokens]
    logger.info(f"过滤掉 {before_filter - len(df)} 行 (token数超过 {max_tokens})，剩余 {len(df)} 行")

    return df


# ==================== Embedding客户端 ====================
class EmbeddingClient:
    """Embedding API客户端，支持批量处理和错误重试"""

    def __init__(self, api_key: str, base_url: str, model: str):
        """
        初始化客户端

        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.retry_count = 3
        self.retry_delay = 2

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成embedding

        Args:
            texts: 文本列表

        Returns:
            embedding向量列表
        """
        for attempt in range(self.retry_count):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model
                )
                return [item.embedding for item in response.data]

            except Exception as e:
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{self.retry_count}): {str(e)}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"API调用最终失败: {str(e)}")
                    raise

    def embed_single(self, text: str) -> List[float]:
        """
        生成单个文本的embedding

        Args:
            text: 文本

        Returns:
            embedding向量
        """
        return self.embed_batch([text])[0]


# ==================== 主处理函数 ====================
def process_embeddings(df: pd.DataFrame, top_n: int = Config.TOP_N,
                       batch_size: int = Config.BATCH_SIZE) -> pd.DataFrame:
    """
    处理数据生成embedding

    Args:
        df: DataFrame
        top_n: 保留最近的N条数据
        batch_size: 批处理大小

    Returns:
        包含embedding的DataFrame
    """
    logger.info("开始处理embedding...")

    # 按时间排序并取最近的N条
    df = df.sort_values('Time').tail(top_n)
    df.drop('Time', axis=1, inplace=True)

    # 初始化客户端
    client = EmbeddingClient(
        api_key=Config.API_KEY,
        base_url=Config.BASE_URL,
        model=Config.EMBEDDING_MODEL
    )

    # 批量处理embedding
    embeddings = []
    texts = df['combined'].tolist()

    logger.info(f"开始批量生成embedding，共 {len(texts)} 条，批大小 {batch_size}")

    for i in tqdm(range(0, len(texts), batch_size), desc="生成Embedding"):
        batch_texts = texts[i:i + batch_size]
        try:
            batch_embeddings = client.embed_batch(batch_texts)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"批处理失败 (索引 {i}): {str(e)}")
            # 对失败的批次进行单条处理
            for text in batch_texts:
                try:
                    embeddings.append(client.embed_single(text))
                except Exception as e2:
                    logger.error(f"单条处理失败: {str(e2)}")
                    embeddings.append(None)

    # 添加embedding列
    df['embedding'] = embeddings

    # 删除token计数列（可选）
    df.drop('count_token', axis=1, inplace=True)

    logger.info(f"Embedding生成完成，成功 {len([e for e in embeddings if e is not None])}/{len(embeddings)} 条")

    return df


# ==================== 保存结果 ====================
def save_results(df: pd.DataFrame, filepath: str):
    """
    保存结果到CSV

    Args:
        df: DataFrame
        filepath: 保存路径
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 保存为CSV
        df.to_csv(filepath, index=False)
        logger.info(f"结果已保存到: {filepath}")

        # 保存embedding统计信息
        embedding_dims = [len(e) for e in df['embedding'] if e is not None]
        if embedding_dims:
            logger.info(f"Embedding维度统计: 最小={min(embedding_dims)}, "
                        f"最大={max(embedding_dims)}, 平均={np.mean(embedding_dims):.0f}")

    except Exception as e:
        logger.error(f"保存失败: {str(e)}")
        raise


# ==================== 主函数 ====================
def main():
    """主执行流程"""
    start_time = time.time()

    try:
        # 1. 验证API密钥
        if not Config.API_KEY:
            logger.error("API密钥未配置，请设置环境变量 DASHSCOPE_API_KEY")
            return

        # 2. 加载和预处理数据
        df = load_and_preprocess_data(Config.INPUT_FILE)

        # 3. 过滤token
        df = filter_by_token_limit(df, Config.MAX_TOKENS)

        # 4. 生成embedding
        df = process_embeddings(df, Config.TOP_N, Config.BATCH_SIZE)

        # 5. 保存结果
        save_results(df, Config.OUTPUT_FILE)

        # 6. 显示统计信息
        elapsed_time = time.time() - start_time
        logger.info(f"处理完成！总耗时: {elapsed_time:.2f}秒")
        logger.info(f"最终数据量: {len(df)} 行")

        # 可选：显示前几行
        print("\n" + "=" * 50)
        print("前3行数据预览:")
        print("=" * 50)
        print(df.head(3)[['ProductId', 'UserId', 'Score']].to_string())

    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}", exc_info=True)


# ==================== 运行 ====================
if __name__ == "__main__":
    main()