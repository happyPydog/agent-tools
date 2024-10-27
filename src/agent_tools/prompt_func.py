"""@prompt"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import update_wrapper
from typing import (
    Any,
    Awaitable,
    Generic,
    Iterable,
    Literal,
    OrderedDict,
    ParamSpec,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    final,
    overload,
)

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from agent_tools.message import (
    AIMessage,
    Message,
    OpenAIMessageType,
    SystemMessage,
    UserMessage,
)
from agent_tools.utilities import get_env_var, split_union_type

P = ParamSpec("P")
R = TypeVar("R", covariant=True)
MessageLikeType: TypeAlias = Union[str, tuple[str, str], Message[Any]]
LLM_Type: TypeAlias = Union[OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI, Any]
OpenAIModel: TypeAlias = Literal["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]


class AsyncPromptFunction(Protocol[P, R]):
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class PromptFunction(Protocol[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class PromptDecorator(Protocol):

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, func: Callable[P, Awaitable[R]]
    ) -> AsyncPromptFunction[P, R]: ...

    @overload
    def __call__(self, func: Callable[P, R]) -> PromptFunction[P, R]: ...


class BaseOpenAIPromptFunction(Generic[P, R]):

    def __init__(
        self,
        name: str,
        parameters: Sequence[inspect.Parameter],
        return_type: type[R],
        messages: Sequence[MessageLikeType],
        llm: LLM_Type | None,
        model_name: OpenAIModel | None = None,
    ) -> None:
        self._name = name
        self._llm = llm
        self._signature = inspect.Signature(
            parameters=parameters,
            return_annotation=return_type,
        )
        self._messages = messages
        self._model_name = model_name
        self._return_types = list(split_union_type(return_type))

    @property
    def return_types(self) -> list[type[R]]:
        return self._return_types.copy()

    @property
    def model_name(self) -> str:
        return self._model_name or self.get_model_name()

    @property
    def llm(self) -> LLM_Type:
        return self._llm or self.get_llm()

    def get_model_name(self) -> str:
        return "gpt-4o"

    def get_llm(self) -> LLM_Type:
        return NotImplemented

    def format(self, *args: P.args, **kwargs: P.kwargs) -> Iterable[OpenAIMessageType]:
        """Format the messages with the given arguments."""
        bound_args = self.get_bound_args(*args, **kwargs)
        messages = []
        for message in self._messages:
            if isinstance(message, str):
                message = UserMessage(message)
            elif isinstance(message, tuple):
                role, text = message[0].lower(), message[1]
                match role:
                    case "human":
                        message = UserMessage(text)
                    case "system":
                        message = SystemMessage(text)
                    case "ai":
                        message = AIMessage(text)
                    case _:
                        raise ValueError(f"Invalid role: '{role}' in message: {message}")
            elif isinstance(message, Message):
                ...
            else:
                raise TypeError(f"Invalid message type: {type(message)}")
            messages.append(message.format(**bound_args))
        return cast(Iterable[OpenAIMessageType], messages)

    def get_bound_args(self, *args: P.args, **kwargs: P.kwargs) -> OrderedDict[str, Any]:
        """Get the bound arguments for the function."""
        bound_args = self._signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return bound_args.arguments


class AsyncOpenAIPromptFunction(BaseOpenAIPromptFunction[P, R], Generic[P, R], AsyncPromptFunction[P, R]):
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Async OpenAI prompt function."""
        messages = self.format(*args, **kwargs)
        chat_completion = await self.llm.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )  # type: ignore[misc]
        return cast(R, chat_completion.choices[0].message.content)

    @final
    def get_llm(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=get_env_var("OPENAI_API_KEY"),
            base_url=get_env_var("OPENAI_BASE_URL"),
        )


class OpenAIPromptFunction(BaseOpenAIPromptFunction[P, R], Generic[P, R], PromptFunction[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """OpenAI prompt function."""
        messages = self.format(*args, **kwargs)
        chat_completion = cast(
            ChatCompletion,
            self.llm.chat.completions.create(
                model=self.model_name,
                messages=messages,
            ),
        )
        return cast(R, chat_completion.choices[0].message.content)

    @final
    def get_llm(self) -> OpenAI:
        return OpenAI(
            api_key=get_env_var("OPENAI_API_KEY"),
            base_url=get_env_var("OPENAI_BASE_URL"),
        )


def openai_prompt(
    *messages: MessageLikeType,
    llm: LLM_Type | None = None,
    model_name: OpenAIModel | None = None,
) -> PromptDecorator:
    def decorator(
        func: Callable[P, Awaitable[R]] | Callable[P, R],
    ) -> AsyncOpenAIPromptFunction[P, R] | OpenAIPromptFunction[P, R]:
        func_signature = inspect.signature(func)
        if inspect.iscoroutinefunction(func):
            async_prompt_func = AsyncOpenAIPromptFunction[P, R](
                name=func.__name__,
                parameters=list(func_signature.parameters.values()),
                return_type=func_signature.return_annotation,
                messages=messages,
                llm=llm,
                model_name=model_name,
            )
            return cast(AsyncOpenAIPromptFunction[P, R], update_wrapper(async_prompt_func, func))

        prompt_func = OpenAIPromptFunction[P, R](
            name=func.__name__,
            parameters=list(func_signature.parameters.values()),
            return_type=func_signature.return_annotation,
            messages=messages,
            llm=llm,
            model_name=model_name,
        )
        return cast(OpenAIPromptFunction[P, R], update_wrapper(prompt_func, func))

    return decorator  # type: ignore[return-value]
