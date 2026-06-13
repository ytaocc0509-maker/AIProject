import asyncio
from contextlib import asynccontextmanager

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from zhipu_ai import llm

mcp_server_config = {
    "url": "http://localhost:8008/sse",
    "transport": "sse"
}


@asynccontextmanager
async def make_agent():
    """生成一个智能体(langgraph)"""
    async with MultiServerMCPClient({'lx_mcp': mcp_server_config}) as client:
        agent = create_react_agent(llm, tools=client.get_tools())
        yield agent


async def main():
    """在异步环境下，创建智能体，并执行"""
    async with make_agent() as agent:
        resp = await agent.ainvoke({'messages': '计算一下(3+6)的结果'})
        print(resp)


if __name__ == '__main__':
    asyncio.run(main())