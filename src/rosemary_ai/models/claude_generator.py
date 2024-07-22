from typing import Generator, Dict, Any, List

from ._option_types import CHAT_OPTION_TYPES
from ._utils import _shape_messages, _update_options, reform_system_message
from .._utils._image import _image_to_data_uri  # noqa
from .generator import AbstractContentGenerator
from anthropic import Anthropic, NOT_GIVEN, AsyncAnthropic

from .._logger import LOGGER


class ClaudeChatGenerator(AbstractContentGenerator[str]):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, data: Dict[str, str | List[Dict[str, str | List]]],
                 options: Dict[str, Any], dry_run: bool) -> str:
        messages = _shape_messages(data.pop('messages'))

        data: Dict[str, List[str]]
        _update_options(options, data, CHAT_OPTION_TYPES)

        messages, system = reform_system_message(messages, 'Claude')
        if system is None:
            system = NOT_GIVEN

        LOGGER.info(f'Sending messages to {self.model_name}: "{system}", "{messages}".')
        LOGGER.debug(f'Options: {options}.')

        api_key = options.pop('api_key', None)

        if 'max_tokens' not in options:
            options['max_tokens'] = 4096

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')
            return ''

        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model=self.model_name, messages=messages,
            **options,
            system=system)

        LOGGER.info(f'Received response from {self.model_name}: "{message.content}".')

        if message.content[0].type == 'tool_use':
            raise NotImplementedError('Tool use in Claude has not been implemented yet.')

        result = message.content[0].text

        return result

    async def generate_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                             options: Dict[str, Any], dry_run: bool) -> str:
        messages = _shape_messages(data.pop('messages'))

        data: Dict[str, List[str]]
        _update_options(options, data, CHAT_OPTION_TYPES)

        messages, system = reform_system_message(messages, 'Claude')
        if system is None:
            system = NOT_GIVEN

        LOGGER.info(f'Sending messages to {self.model_name}: "{system}", "{messages}".')
        LOGGER.debug(f'Options: {options}.')

        api_key = options.pop('api_key', None)

        if 'max_tokens' not in options:
            options['max_tokens'] = 4096

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')
            return ''

        client = AsyncAnthropic(api_key=api_key)
        message = await client.messages.create(
            model=self.model_name, messages=messages,
            **options,
            system=system)

        LOGGER.info(f'Received response from {self.model_name}: "{message.content}".')

        if message.content[0].type == 'tool_use':
            raise NotImplementedError('Tool use in Claude has not been implemented yet.')

        result = message.content[0].text

        return result

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]],
                        options: Dict[str, Any], dry_run: bool) -> Generator[str, None, None]:
        messages = _shape_messages(data.pop('messages'))

        data: Dict[str, List[str]]
        _update_options(options, data, CHAT_OPTION_TYPES)

        messages, system = reform_system_message(messages, 'Claude')
        if system is None:
            system = NOT_GIVEN

        LOGGER.info(f'Sending messages to {self.model_name}: "{system}", "{messages}".')
        LOGGER.debug(f'Options: {options}.')

        api_key = options.pop('api_key', None)

        if 'max_tokens' not in options:
            options['max_tokens'] = 4096

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')
            return

        client = Anthropic(api_key=api_key)
        with client.messages.stream(model=self.model_name, messages=messages,
                                    **options,
                                    system=system) as completion_stream:
            result = ''

            for chunk in completion_stream.text_stream:
                LOGGER.info(f'Received response (streaming) from {self.model_name}: "{chunk}".')

                if chunk is not None:
                    result += chunk
                    yield result

    async def generate_stream_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                                    options: Dict[str, Any], dry_run: bool) -> Generator[str, None, None]:
        messages = _shape_messages(data.pop('messages'))

        data: Dict[str, List[str]]
        _update_options(options, data, CHAT_OPTION_TYPES)

        messages, system = reform_system_message(messages, 'Claude')
        if system is None:
            system = NOT_GIVEN

        LOGGER.info(f'Sending messages to {self.model_name}: "{system}", "{messages}".')
        LOGGER.debug(f'Options: {options}.')

        api_key = options.pop('api_key', None)

        if 'max_tokens' not in options:
            options['max_tokens'] = 4096

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')
            return

        client = AsyncAnthropic(api_key=api_key)
        async with client.messages.stream(model=self.model_name, messages=messages,
                                          **options,
                                          system=system) as completion_stream:
            result = ''

            async for chunk in completion_stream.text_stream:
                LOGGER.info(f'Received response (streaming) from {self.model_name}: "{chunk}".')

                if chunk is not None:
                    result += chunk
                    yield result
