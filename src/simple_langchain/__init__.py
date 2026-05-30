"""
Simple LangChain — 一个简化版 LangChain 实现

和 simple_langgraph 在同一个仓库里：
- LangGraph 关注**状态编排**（图、节点、边、循环）
- LangChain 关注**组件组合**（prompt、llm、parser、chain、memory、retriever、agent）
"""

from simple_langchain.prompts import PromptTemplate, ChatPromptTemplate
from simple_langchain.llms import BaseLLM, FakeListLLM, Generation, LLMResult
from simple_langchain.parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser,
)
from simple_langchain.chains import LLMChain, SequentialChain, RetrievalChain
from simple_langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
)
from simple_langchain.retrievers import Document, CharacterTextSplitter, SimpleVectorStore
from simple_langchain.agents import Tool, AgentExecutor
