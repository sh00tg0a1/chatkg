from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent

import openai
import os
import dotenv

env_file = '../../.env'
dotenv.load_dotenv(env_file, override=True)


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_ENDPOINT  = os.environ.get("OPENAI_ENDPOINT")
OPENAI_API_VERSION  = os.environ.get("OPENAI_API_VERSION")
OPENAI_API_TYPE  = os.environ.get("OPENAI_API_TYPE")

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

serpapi_api_key=os.environ.get("SERPAPI_API_KEY")

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