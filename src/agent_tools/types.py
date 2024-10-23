"""Types."""

from enum import Enum
from typing import Any, Iterable, TypedDict, Union

from langchain_openai.chat_models import AzureChatOpenAI as LangchainAzureChatOpenAI
from langchain_openai.chat_models import ChatOpenAI as LangchainChatOpenAI
from openai import AzureOpenAI, OpenAI


class Role(Enum):
    SYSTEM = "system"
    HUMAN = "human"
    AI = "ai"


class Message(TypedDict):
    role: Role
    content: str


class LangfuseCallbackConfig(TypedDict, total=False):
    public_key: str | None
    secret_key: str | None
    host: str | None
    debug: bool
    update_stateful_client: bool
    session_id: str | None
    user_id: str | None
    trace_name: str | None
    release: str | None
    version: str | None
    metadata: dict[str, Any] | None
    tags: list[str] | None
    threads: int | None
    flush_at: int | None
    flush_interval: int | None
    max_retries: int | None
    timeout: int | None
    enabled: bool | None
    sdk_integration: str | None
    sample_rate: float | None


ModelType = Union[LangchainChatOpenAI, LangchainAzureChatOpenAI, OpenAI, AzureOpenAI]
MessageLikeType = Union[str, Iterable[Union[tuple[Role, str], tuple[str, str], Message]]]
