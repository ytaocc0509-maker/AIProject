import asyncio
from typing import TypedDict, Annotated

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.constants import END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import create_react_agent

from zhipu_ai import llm

mcp_server_config = {
    "url": "http://localhost:8008/sse",
    "transport": "sse"
}


class MyState(TypedDict):
    email: str
    messages: Annotated[list, add_messages]


async def async_node(state: MyState):
    """定义一个节点（调用MCP服务端的工具）"""
    print(state.get('email'))
    async with MultiServerMCPClient({'lx_mcp': mcp_server_config}) as client:
        agent = create_react_agent(llm, tools=client.get_tools())
        resp = await agent.ainvoke(state)
        return resp


async def async_resource(state: MyState):
    """定义一个节点（加载MCP服务端的资源）"""

    async with MultiServerMCPClient({'lx_mcp': mcp_server_config}) as client:
        resource = await client.get_resources('lx_mcp', uris='datas://users/567/email')

        return {'email': resource[0].model_dump().get('data')}


workflow = StateGraph(MyState)

workflow.add_node('agent', async_node)
workflow.add_node('resource', async_resource)

workflow.set_entry_point('resource')
workflow.add_edge('resource', 'agent')
workflow.add_edge('agent', END)

# 编译得到：异步的工作流
graph = workflow.compile()

_printed = set()


async def execute_graph():
    """执行该 工作流"""
    while True:
        user_input = input("用户：")
        if user_input.lower() in ['q', 'exit', 'quit']:
            print('对话结束，拜拜！')
            break
        else:
            async for event in graph.astream({"messages": [("user", user_input)]}, stream_mode="values"):
                _print_event(event, _printed)


def _print_event(event: dict, _printed: set, max_length=1500):
    """
    打印事件信息，特别是对话状态和消息内容。如果消息内容过长，会进行截断处理以保证输出的可读性。

    参数:
        event (dict): 事件字典，包含对话状态和消息。
        _printed (set): 已打印消息的集合，用于避免重复打印。
        max_length (int): 消息的最大长度，超过此长度将被截断。默认值为1500。
    """
    current_state = event.get("dialog_state")
    if current_state:
        print("当前处于: ", current_state[-1])  # 输出当前的对话状态
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]  # 如果消息是列表，则取最后一个
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... （已截断）"  # 超过最大长度则截断
            print(msg_repr)  # 输出消息的表示形式
            _printed.add(message.id)  # 将消息ID添加到已打印集合中


if __name__ == '__main__':
    asyncio.run(execute_graph())
