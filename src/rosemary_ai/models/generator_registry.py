from typing import List

from . import _model_info

from .claude_generator import ClaudeChatGenerator
from .cohere_generator import CohereChatGenerator
from .generator import AbstractContentGenerator
from .gpt_generator import GPTChatGenerator, GPTImageGenerator


_MODEL_GENERATORS = {}


def get_generator(model_name: str) -> AbstractContentGenerator | None:
    if not model_name:
        return None

    if model_name not in _MODEL_GENERATORS:
        raise ValueError(f"Model generator for '{model_name}' not found.")

    return _MODEL_GENERATORS[model_name]


def register_generator(model_name: str | List[str], generator: AbstractContentGenerator):
    if isinstance(model_name, list):
        for name in model_name:
            register_generator(name, generator)
    else:
        _MODEL_GENERATORS[model_name] = generator


def generator_list() -> list[str]:
    return list(_MODEL_GENERATORS.keys())


# OpenAI
for model_name, in_lib_names in _model_info.GPT.items():
    register_generator(in_lib_names, GPTChatGenerator(model_name))

for model_name, in_lib_names in _model_info.DALL_E.items():
    register_generator(in_lib_names, GPTImageGenerator(model_name))


# Anthropic
for model_name, in_lib_names in _model_info.CLAUDE.items():
    register_generator(in_lib_names, ClaudeChatGenerator(model_name))


# Cohere
for model_name, in_lib_names in _model_info.COMMAND.items():
    register_generator(in_lib_names, CohereChatGenerator(model_name))

