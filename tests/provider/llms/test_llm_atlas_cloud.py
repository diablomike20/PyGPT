#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pygpt_net.core.types.reasoning import PROVIDER_REASONING_EFFORTS
from pygpt_net.item.model import ModelItem
from pygpt_net.provider.llms.atlas_cloud.provider import AtlasCloudLLM

BASE_URL = "https://api.atlascloud.ai/v1"


def _window(api_key="CONFIG-KEY", base_url=BASE_URL):
    config = MagicMock()
    config.get.return_value = None
    return SimpleNamespace(
        core=SimpleNamespace(
            config=config,
            api=SimpleNamespace(logger=MagicMock()),
            models=SimpleNamespace(
                prepare_client_args=MagicMock(
                    return_value={"api_key": api_key, "base_url": base_url}
                )
            ),
        )
    )


def test_setup_is_openai_compatible_with_atlas_defaults():
    setup = AtlasCloudLLM().setup()

    assert setup["openai_compatible"] is True
    assert setup["settings"]["api_base"]["default"] == BASE_URL
    assert setup["settings"]["api_key"]["secret"] is True


def test_chat_only_provider():
    provider = AtlasCloudLLM()

    assert provider.id == "atlas_cloud"
    assert provider.type == ["llama_index"]
    assert "atlas_cloud" in PROVIDER_REASONING_EFFORTS


def test_llama_uses_provider_settings():
    provider = AtlasCloudLLM()
    provider.inject_llamaindex_http_clients = MagicMock(side_effect=lambda args, cfg: args)
    model = ModelItem("qwen/qwen3.5-flash")
    model.provider = "atlas_cloud"
    model.tool_calls = True
    model.llama_index = {"args": []}

    with patch("llama_index.llms.openai_like.OpenAILike", return_value="LLM") as openai_like:
        result = provider.llama(_window(), model)

    assert result == "LLM"
    kwargs = openai_like.call_args.kwargs
    assert kwargs["model"] == "qwen/qwen3.5-flash"
    assert kwargs["api_key"] == "CONFIG-KEY"
    assert kwargs["api_base"] == BASE_URL
    assert kwargs["is_chat_model"] is True
    assert kwargs["is_function_calling_model"] is True


def test_environment_names_are_atlascloud_first():
    keys, bases = AtlasCloudLLM().get_openai_compatible_env_names()

    assert keys[0] == "ATLASCLOUD_API_KEY"
    assert bases[0] == "ATLASCLOUD_API_BASE"
