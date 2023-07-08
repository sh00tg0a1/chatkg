from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent

import openai

OPENAI_API_KEY = "5cc0b8aef94b47db86c5b8c7f61ab9f2"
OPENAI_ENDPOINT  = "https://anyshare-demo-chatgpt.openai.azure.com/"
OPENAI_API_VERSION  = "2023-03-15-preview"
OPENAI_API_TYPE  = "azure"

openai.api_type = OPENAI_API_TYPE
openai.api_base = OPENAI_ENDPOINT
openai.api_version = OPENAI_API_VERSION
openai.api_key = OPENAI_API_KEY

llm = OpenAI(
    # engine = "asada001",
    # engine="aschatgpt35",
    model_kwargs= {
        "engine": "asdavinci003",
    },
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_ENDPOINT,
    max_tokens=256,
)

serpapi_api_key='e457fc03ca3c0fc5c94958e91bf2259d5142197587ab0cb2ce2479bfc0be5698'

search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
tools = [
    Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history")

agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

agent_chain.run(input="hi, i am bob")