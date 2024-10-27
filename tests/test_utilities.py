"""Test the utilities helper functions."""

import pytest

from agent_tools.utilities import get_env_var, is_openai_model_type


def test_get_env_var(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", "/home/runner")
    monkeypatch.setenv("USER", "runner")
    monkeypatch.setenv("SHELL", "/bin/bash")

    assert get_env_var("HOME") == "/home/runner"
    assert get_env_var("USER") == "runner"
    assert get_env_var("SHELL") == "/bin/bash"


def test_is_openai_model_type(monkeypatch: pytest.MonkeyPatch):
    from langfuse.openai import AzureOpenAI as LangFuseAzureOpenAI
    from langfuse.openai import OpenAI as LangFuseOpenAI
    from openai import AzureOpenAI, OpenAI

    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_api_key")
    monkeypatch.setenv("OPENAI_BASE_URL", "test_openai_base_url")
    monkeypatch.setenv("OPENAI_API_VERSION", "test_openai_api_version")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_azure_openai_api_key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "test_azure_openai_endpoint")

    assert is_openai_model_type(OpenAI)
    assert is_openai_model_type(OpenAI())
    assert is_openai_model_type(LangFuseOpenAI)
    assert is_openai_model_type(LangFuseOpenAI())

    assert is_openai_model_type(AzureOpenAI)
    assert is_openai_model_type(AzureOpenAI())
    assert is_openai_model_type(LangFuseAzureOpenAI)
    assert is_openai_model_type(LangFuseAzureOpenAI())
