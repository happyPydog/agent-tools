"""Message."""

from typing import Generic, TypeVar

R = TypeVar("R")


class Message(Generic[R]):
    pass


class AIMessage(Message[R]):
    pass


class HumanMessage(Message[R]):
    pass


class SystemMessage(Message[R]):
    pass


class FunctionMessage(Message[R]):
    pass
