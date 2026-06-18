from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, MessagesPlaceholder

from langchain_demo.my_llm import llm

from langchain_core.prompts import ChatPromptTemplate


# {topic} : 变量占位符
#   消息占位符
# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", "你是一个幽默的电视台主持人！"),
#     ("user", "帮我生成一个简短的，关于{topic}的报幕词。")
# ])
# print(prompt_template.invoke({"topic": "相声"}))


prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个幽默的电视台主持人！"),
    MessagesPlaceholder("input")
])

prompt_template.invoke({"input": [HumanMessage(content="你好，主持人!")]})


chain = prompt_template | llm

print(chain.invoke({"input": [HumanMessage(content="你好，主持人!")]}))