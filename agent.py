from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_pinecone import PineconeEmbeddings

from deepagents import create_deep_agent

from config import CHAT_MODEL, DB_FOLDER, VISION_MODEL

vision_llm = ChatOllama(model=VISION_MODEL, temperature=0)

embeddings = PineconeEmbeddings()

collection = Chroma(
    collection_name="video-assistant",
    embedding_function=embeddings,
    persist_directory=str(DB_FOLDER),
)

prompt_template = ChatPromptTemplate.from_template(
    """
    You are a CamQuery Assistant.

    Use:
    1. Retrieved video memory
    2. Previous conversation context

    to answer naturally and consistently.

    =============================
    PREVIOUS CONVERSATION
    =============================
    {conversation_context}

    =============================
    RETRIEVED VIDEO MEMORY
    =============================
    {memory}

    =============================
    CURRENT USER QUESTION
    =============================
    {message}

    Instructions:
    - Use ONLY retrieved frame/video data
    - Reference timestamps when possible
    - Understand follow-up questions using chat history
    - If information is unavailable, say so clearly
    """
)


def create_chat_runnable() -> RunnableSequence:
    chat_agent = create_deep_agent(model=f"ollama:{CHAT_MODEL}", tools=[])
    return RunnableSequence(prompt_template, chat_agent)
