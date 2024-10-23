"""Utilities."""

from langchain_openai.chat_models import AzureChatOpenAI as LangchainAzureChatOpenAI
from langchain_openai.chat_models import ChatOpenAI as LangchainChatOpenAI
from openai import AzureOpenAI, OpenAI

from agent_tools.types import ModelType


def env_error(env_var: str) -> None:
    raise ValueError(f"{env_var} environment variable is not set.")


def is_openai_model(model: ModelType) -> bool:
    return isinstance(model, (OpenAI, AzureOpenAI))


def is_langchain_model(model: ModelType) -> bool:
    return isinstance(model, (LangchainChatOpenAI, LangchainAzureChatOpenAI))
