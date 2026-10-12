#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ================================================== #
# This file is a part of PYGPT package               #
# Website: https://pygpt.net                         #
# GitHub:  https://github.com/szczyglis-dev/py-gpt   #
# MIT License                                        #
# ================================================== #

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_index.core.llms.llm import BaseLLM as LlamaBaseLLM

from pygpt_net.core.types import MODE_LLAMA_INDEX
from pygpt_net.provider.llms.base import BaseLLM
from pygpt_net.item.model import ModelItem


class AtlasCloudLLM(BaseLLM):
    def __init__(self, *args, **kwargs):
        super(AtlasCloudLLM, self).__init__(*args, **kwargs)
        self.id = "atlas_cloud"
        self.name = "Atlas Cloud"
        # Atlas Cloud has no embeddings endpoint, so only chat models are exposed.
        self.type = [MODE_LLAMA_INDEX]

    def setup(self) -> dict:
        from .config import setup
        return setup()

    def llama(
            self,
            window,
            model: ModelItem,
            stream: bool = False
    ) -> LlamaBaseLLM:
        """
        Return LlamaIndex chat provider

        :param window: window instance
        :param model: model instance
        :param stream: stream mode
        :return: LLM provider instance
        """
        from llama_index.llms.openai_like import OpenAILike
        args = self.prepare_openai_compatible_args(window, model)
        if "is_chat_model" not in args:
            args["is_chat_model"] = True
        if "is_function_calling_model" not in args:
            args["is_function_calling_model"] = model.tool_calls
        args = self.inject_llamaindex_http_clients(args, window.core.config)
        self.log_llama_create(window, model, args, "OpenAILike")
        return OpenAILike(**args)
