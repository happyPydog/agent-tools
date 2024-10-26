"""@prompt"""

from collections.abc import Callable
from typing import Awaitable, Generic, ParamSpec, Protocol, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R", covariant=True)


class AsyncPromptFunction(Protocol[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Awaitable[R]: ...


class PromptFunction(Protocol[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class PromptDecorator(Protocol):

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, func: Callable[P, Awaitable[R]]
    ) -> AsyncPromptFunction[P, R]: ...

    @overload
    def __call__(self, func: Callable[P, R]) -> PromptFunction[P, R]: ...


class BaseOpenAIPromptFunction(Generic[R]):
    pass


class AsyncOpenAIPromptFunction(BaseOpenAIPromptFunction[Awaitable[R]]):
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Async OpenAI prompt function."""
        return NotImplemented


class OpenAIPromptFunction(BaseOpenAIPromptFunction[R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """OpenAI prompt function."""
        return NotImplemented
