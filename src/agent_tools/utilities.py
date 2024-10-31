"""Utilities helper functions."""

import json
import os
import re
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


def extract_json_content(text: str) -> list[dict[str, str]] | None | Any:
    """
    Extract JSON content enclosed within ```json ... ``` from the given text.

    This function searches for a JSON block in the provided text that is enclosed within triple backticks
    and labeled as 'json'. It cleans the JSON text by removing unescaped newline characters within strings,
    which are invalid in JSON. The cleaned JSON text is then parsed and returned as a list of dictionaries.

    Args:
        text (str): The input text containing the JSON block.

    Returns:
        Optional[List[Dict[str, str]]]: The parsed JSON content as a list of dictionaries,
        or None if no valid JSON block is found.
    """
    json_text = _extract_json_block(text)
    if json_text is None:
        return None

    cleaned_json_text = _clean_json_text(json_text)

    try:
        return json.loads(cleaned_json_text)
    except json.JSONDecodeError:
        return None


def _extract_json_block(text: str) -> str | None:
    """
    Extract the JSON block labeled with ```json from the text.

    Args:
        text (str): The input text.

    Returns:
        Optional[str]: The extracted JSON text, or None if not found.
    """
    pattern = re.compile(r"```json\s*(\[[\s\S]*?\])\s*```", re.MULTILINE)
    match = pattern.search(text)
    return match.group(1) if match else None


def _clean_json_text(json_text: str) -> str:
    """
    Clean the JSON text by removing unescaped newlines within strings.

    Args:
        json_text (str): The raw JSON text.

    Returns:
        str: The cleaned JSON text suitable for parsing.
    """
    # Regex pattern to find all string literals in JSON
    string_pattern = re.compile(r"\"(.*?)(?<!\\)\"", re.DOTALL)

    def replace_newlines(match: re.Match) -> str:
        # Replace unescaped newlines within strings with spaces
        string_content = match.group(1).replace("\n", " ")
        return f'"{string_content}"'

    return string_pattern.sub(replace_newlines, json_text)
