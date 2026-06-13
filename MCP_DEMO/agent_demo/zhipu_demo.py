from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from zhipu_ai import zhipuai_client, llm


class SearchInput(BaseModel):
    query: str = Field(description='需要搜索的内容或者关键词')


@tool('my_search_tool', args_schema=SearchInput, description='专门搜索互联网中的内容')
def my_search(query: str) -> str:
    """搜索互联网上的内容"""
    try:
        response = zhipuai_client.web_search.web_search(
            search_engine="search-std",
            search_query=query
        )
        print(response)
        if response.search_result:
            return "\n\n".join([d.content for d in response.search_result])
    except Exception as e:
        print(e)
        return '没有搜索到任何内容！'


prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个智能助手，尽可能的调用工具回答用户的问题'),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    ('human', '{input}'),
    MessagesPlaceholder(variable_name='agent_scratchpad', optional=True),
])

tools = [my_search]
agent = create_tool_calling_agent(llm, tools, prompt)

executor = AgentExecutor(agent=agent, tools=[my_search])
#
# resp = executor.invoke({'input': '今天，长沙的天气情况'})
# print(resp)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


agent_with_history = RunnableWithMessageHistory(
    executor,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)

resp1 = agent_with_history.invoke(
    {'input': '你好， 我是LAOXIAO，出生在1983年'},
    config={'configurable': {'session_id': 'lx111'}}
)

print(resp1)

resp2 = agent_with_history.invoke(
    {'input': '我出生的那一年，国际上发生哪些大事'},
    config={'configurable': {'session_id': 'lx111'}}
)

print(resp2)