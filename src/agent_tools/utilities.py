"""Utilities helper functions."""

import os
from typing import Any

from openai import OpenAI
from typing_extensions import TypeIs


def get_env_var(name: str) -> str:
    if name in os.environ:
        return os.environ[name]
    raise ValueError(f"Environment variable '{name}' is not set.")


def is_openai_model_type(llm: Any) -> TypeIs[OpenAI]:
    return isinstance(llm, OpenAI) or issubclass(llm, OpenAI)


if __name__ == "__main__":
    print(get_env_var("HOME"))
    print(is_openai_model_type(OpenAI))
    print(is_openai_model_type(OpenAI()))

    from langfuse.openai import OpenAI as LangFuseOpenAI

    print(is_openai_model_type(LangFuseOpenAI))
