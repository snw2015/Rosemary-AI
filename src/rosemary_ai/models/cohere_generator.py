from typing import Generator, Dict, Any, List, Tuple

from ._option_types import CHAT_OPTION_TYPES
from ._utils import shape_messages, update_options, reform_system_message
from .generator import AbstractContentGenerator

from .._logger import LOGGER
from cohere import Client, AsyncClient

_ROLE_TABLE = {
    'user': 'USER',
    'system': 'SYSTEM',
    'assistant': 'ASSISTANT'
}


def _convert_to_cohere_message(messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
    cohere_messages = []
    for message in messages:
        role = _ROLE_TABLE[message['role']]
        content = message['content']
        cohere_messages.append({'role': role, 'message': content})

    if cohere_messages and cohere_messages[-1]['role'] == 'USER':
        last_message = cohere_messages.pop()['message']
    else:
        last_message = ' '

    return cohere_messages, last_message


class CohereChatGenerator(AbstractContentGenerator[str]):
    def __init__(self, model_name: str):
        super().__init__('Cohere')
        self.model_name = model_name

    def _set_up(self, data: Dict[str, str | List[Dict[str, str | List]]],
                options: Dict[str, Any], dry_run: bool, api_key: str) -> Tuple:
        messages = shape_messages(data.pop('messages'))

        data: Dict[str, List[str]]
        update_options(options, data, CHAT_OPTION_TYPES)

        LOGGER.info(f'Sending messages to {self.model_name}: "{messages}".')
        LOGGER.debug(f'Options: {options}.')

        messages, system = reform_system_message(messages, 'Cohere')
        messages, last_message = _convert_to_cohere_message(messages)

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')

        api_key = self.get_api_key(api_key)

        return messages, last_message, system, options, api_key

    def generate(self, data: Dict[str, str | List[Dict[str, str | List]]],
                 options: Dict[str, Any],
                 dry_run: bool, api_key: str = None) -> str:
        messages, last_message, system, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return ''

        client = Client(api_key=api_key)
        message = client.chat(
            model=self.model_name,
            message=last_message,
            preamble=system,
            **options
        )

        LOGGER.info(f'Received response from {self.model_name}: "{message}".')

        result = message.text

        return result

    async def generate_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                             options: Dict[str, Any],
                             dry_run: bool, api_key: str = None) -> str:
        messages, last_message, system, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return ''

        client = AsyncClient(api_key=api_key)
        message = await client.chat(
            model=self.model_name,
            message=last_message,
            preamble=system,
            **options
        )

        LOGGER.info(f'Received response from {self.model_name}: "{message}".')

        result = message.text

        return result

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]],
                        options: Dict[str, Any],
                        dry_run: bool, api_key: str = None) -> Generator[str, None, None]:
        messages, last_message, system, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return

        client = Client(api_key=api_key)

        result = ''

        for chunk in client.chat_stream(
                model=self.model_name,
                message=last_message,
                preamble=system,
                **options
        ):
            LOGGER.info(f'Received response (streaming) from {self.model_name}: "{chunk}".')

            if chunk.event_type != 'text-generation':
                continue

            if chunk is not None:
                result += chunk.text
                yield result

    async def generate_stream_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                                    options: Dict[str, Any],
                                    dry_run: bool, api_key: str = None) -> Generator[str, None, None]:
        messages, last_message, system, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return

        client = AsyncClient(api_key=api_key)

        result = ''

        async for chunk in client.chat_stream(
                model=self.model_name,
                message=last_message,
                preamble=system,
                **options
        ):
            LOGGER.info(f'Received response (streaming) from {self.model_name}: "{chunk}".')

            if chunk.event_type != 'text-generation':
                continue

            if chunk is not None:
                result += chunk.text
                yield result
