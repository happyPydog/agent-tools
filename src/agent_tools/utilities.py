"""Utilities helper functions."""

import os
import types
from functools import wraps
from typing import (
    Any,
    Callable,
    ParamSpec,
    Sequence,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from openai import OpenAI
from pydantic import BaseModel
from typing_extensions import TypeIs

P = ParamSpec("P")
R = TypeVar("R")
TypeT = type[R]


def get_env_var(name: str) -> str:
    if name in os.environ:
        return os.environ[name]
    raise ValueError(f"Environment variable '{name}' is not set.")


def is_openai_model_type(llm: Any) -> TypeIs[OpenAI]:
    return isinstance(llm, OpenAI) or issubclass(llm, OpenAI)


def is_union_type(type_: type) -> bool:
    """Return True if the type is a union type."""
    type_ = get_origin(type_) or type_
    return type_ is Union or type_ is types.UnionType


def is_builtin_type(type_: type) -> bool:
    """Return True if the type is a builtin type."""
    return type_.__module__ == "builtins"


def is_pydantic_model_type(type_: type) -> bool:
    """Return True if the type is a Pydantic model type."""
    return issubclass(type_, BaseModel)


def split_union_type(type_: TypeT) -> Sequence[TypeT]:
    """Split a union type into its constituent types."""
    return get_args(type_) if is_union_type(type_) else [type_]


def discard_none_arguments(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to discard function arguments with value `None`"""

    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        non_none_kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return func(*args, **non_none_kwargs)  # type: ignore[arg-type]

    return wrapped
