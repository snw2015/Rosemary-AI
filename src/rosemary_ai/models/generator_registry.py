from typing import List

from .generator import AbstractContentGenerator
from .gpt_generator import GPTChatGenerator, GPTImageGenerator

_model_generators = {}


def get_generator(model_name: str) -> AbstractContentGenerator | None:
    if not model_name:
        return None

    if model_name not in _model_generators:
        raise ValueError(f"Model generator for '{model_name}' not found.")

    return _model_generators[model_name]


def register_generator(model_name: str | List[str], generator: AbstractContentGenerator):
    if isinstance(model_name, list):
        for name in model_name:
            register_generator(name, generator)
    else:
        _model_generators[model_name] = generator


def generator_list() -> list[str]:
    return list(_model_generators.keys())


register_generator(['gpt-3.5-t', 'gpt-3.5-turbo'], GPTChatGenerator('gpt-3.5-turbo'))
register_generator(['gpt-4-t', 'gpt-4-turbo'], GPTChatGenerator('gpt-4-turbo'))
register_generator('gpt-4o', GPTChatGenerator('gpt-4o'))
register_generator('dall-e-3', GPTImageGenerator('dall-e-3'))

