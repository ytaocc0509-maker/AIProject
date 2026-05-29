from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


# 状态类
class State(TypedDict):
    """
    定义状态字典的结构。

    参数:
        messages (list[AnyMessage]): 消息列表。
        user_info (str): 用户信息。
    """
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
