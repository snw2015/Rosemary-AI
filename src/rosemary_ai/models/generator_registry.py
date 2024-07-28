from typing import List

from . import _model_info

from .claude_generator import ClaudeChatGenerator
from .cohere_generator import CohereChatGenerator
from .generator import AbstractContentGenerator
from .gpt_generator import GPTChatGenerator, GPTImageGenerator, GPTEmbeddingGenerator
from .stability_generator import StabilityImageGenerator, StabilityV1ImageGenerator

_MODEL_GENERATORS = {}


def get_generator(model_name_: str) -> AbstractContentGenerator | None:
    if not model_name_:
        return None

    if model_name_ not in _MODEL_GENERATORS:
        raise ValueError(f"Model generator for '{model_name_}' not found.")

    return _MODEL_GENERATORS[model_name_]


def register_generator(model_name_: str | List[str], generator: AbstractContentGenerator):
    if isinstance(model_name_, list):
        for name in model_name_:
            register_generator(name, generator)
    else:
        _MODEL_GENERATORS[model_name_] = generator


def generator_list() -> list[str]:
    return list(_MODEL_GENERATORS.keys())


# OpenAI
for formal_model_name, in_lib_names in _model_info.GPT.items():
    register_generator(in_lib_names, GPTChatGenerator(formal_model_name))

for formal_model_name, in_lib_names in _model_info.DALL_E.items():
    register_generator(in_lib_names, GPTImageGenerator(formal_model_name))

for formal_model_name, in_lib_names in _model_info.OPENAI_EMBEDDINGS.items():
    register_generator(in_lib_names, GPTEmbeddingGenerator(formal_model_name))


# Anthropic
for formal_model_name, in_lib_names in _model_info.CLAUDE.items():
    register_generator(in_lib_names, ClaudeChatGenerator(formal_model_name))


# Cohere
for formal_model_name, in_lib_names in _model_info.COMMAND.items():
    register_generator(in_lib_names, CohereChatGenerator(formal_model_name))


# Stability
for formal_model_name, in_lib_names in _model_info.STABLE_GEN_V2.items():
    register_generator(in_lib_names, StabilityImageGenerator(formal_model_name))

for formal_model_name, in_lib_names in _model_info.STABLE_GEN_V1.items():
    register_generator(in_lib_names, StabilityV1ImageGenerator(formal_model_name))
