import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY

MODEL_NAME = "gpt-4o-mini"

llm = ChatOpenAI(
    temperature=0.0,
    model=MODEL_NAME,
    openai_api_base="https://api.vsegpt.ru/v1",
    openai_api_key=OPENAI_API_KEY,
)

llm4o = ChatOpenAI(
    temperature=0.0,
    model="gpt-4o",
    openai_api_base="https://api.vsegpt.ru/v1",
    openai_api_key=OPENAI_API_KEY,
)

embed_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_base="https://api.vsegpt.ru/v1",
    openai_api_key=OPENAI_API_KEY
)
