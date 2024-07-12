from typing import Generator, Dict, Any, List

from ._utils import _shape_messages
from .._utils._image import _image_to_data_uri  # noqa
from .generator import AbstractContentGenerator
from anthropic import Anthropic, NOT_GIVEN

from .._logger import LOGGER


class ClaudeChatGenerator(AbstractContentGenerator):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, data: Dict[str, str | List[Dict[str, str | List]]], options: Dict[str, Any]) -> str:
        messages = _shape_messages(data['messages'])

        LOGGER.info(f'Sending messages to {self.model_name}: "{messages}".')

        api_key = options.get('api_key')
        temperature = options.get('temperature', NOT_GIVEN)
        max_tokens = options.get('max_tokens', 4096)

        if messages[0]['role'] == 'system':
            system = messages[0]['content']
            messages = messages[1:]
        else:
            system = NOT_GIVEN

        client = Anthropic(api_key=api_key) if api_key else Anthropic()
        message = client.messages.create(
            model=self.model_name, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            system=system)

        LOGGER.info(f'Received response from {self.model_name}: "{message.content}".')

        if message.content[0].type == 'tool_use':
            raise NotImplementedError('Tool use in Claude has not been implemented yet.')

        result = message.content[0].text

        return result

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]],
                        options: Dict[str, Any]) -> Generator[str, None, None]:
        messages = _shape_messages(data['messages'])

        LOGGER.info(f'Sending messages to {self.model_name}: "{messages}".')

        api_key = options.get('api_key')
        temperature = options.get('temperature', NOT_GIVEN)
        max_tokens = options.get('max_tokens', 4096)

        if messages[0]['role'] == 'system':
            system = messages[0]['content']
            messages = messages[1:]
        else:
            system = NOT_GIVEN

        client = Anthropic(api_key=api_key)
        with client.messages.stream(model=self.model_name, messages=messages,
                                    temperature=temperature, max_tokens=max_tokens,
                                    system=system) as completion_stream:
            result = ''

            for chunk in completion_stream.text_stream:
                LOGGER.info(f'Received response (streaming) from {self.model_name}: "{chunk}".')

                if chunk is not None:
                    result += chunk
                    yield result
