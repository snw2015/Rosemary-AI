from typing import List

from .claude_generator import ClaudeChatGenerator
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


register_generator(['gpt-3.5-t', 'gpt-3.5-turbo'], GPTChatGenerator('gpt-3.5-turbo'))
register_generator('gpt-4', GPTChatGenerator('gpt-4'))
register_generator(['gpt-4-t', 'gpt-4-turbo'], GPTChatGenerator('gpt-4-turbo'))
register_generator('gpt-4o', GPTChatGenerator('gpt-4o'))
register_generator(['claude-3.5-s', 'claude-3.5-sonnet'], ClaudeChatGenerator('claude-3-5-sonnet-20240620'))
register_generator(['claude-3-h', 'claude-3-haiku'], ClaudeChatGenerator('claude-3-haiku-20240307'))
register_generator(['claude-3-s', 'claude-3-sonnet'], ClaudeChatGenerator('claude-3-sonnet-20240229'))
register_generator(['claude-3-o', 'claude-3-opus'], ClaudeChatGenerator('claude-3-opus-20240229'))

register_generator('dall-e-3', GPTImageGenerator('dall-e-3'))
