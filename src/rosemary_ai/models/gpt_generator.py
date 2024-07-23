from typing import Generator, Dict, Any, List, Tuple

from ._option_types import CHAT_OPTION_TYPES, GPT_IMAGE_OPTION_TYPES
from ._utils import shape_messages, update_options
from ..exceptions import RmlFormatException
from .generator import AbstractContentGenerator
from openai import OpenAI, AsyncOpenAI

from .._logger import LOGGER


class GPTChatGenerator(AbstractContentGenerator[str]):
    def __init__(self, model_name: str):
        super().__init__('OpenAI')
        self.model_name = model_name

    def _set_up(self, data: Dict[str, str | List[Dict[str, str | List]]],
                options: Dict[str, Any], dry_run: bool, api_key: str) -> Tuple:
        messages = shape_messages(data.pop('messages'))

        data: Dict[str, List[str]]
        update_options(options, data, CHAT_OPTION_TYPES)

        LOGGER.info(f'Sending messages to {self.model_name}: "{messages}".')
        LOGGER.debug(f'Options: {options}.')

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')

        api_key = self.get_api_key(api_key)

        return messages, options, api_key

    def generate(self, data: Dict[str, str | List[Dict[str, str | List]]],
                 options: Dict[str, Any], dry_run: bool, api_key: str = None) -> str:
        messages, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return ''

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=self.model_name, messages=messages,
            **options)

        LOGGER.info(f'Received response from {self.model_name}: "{completion.choices[0].message}".')

        result = completion.choices[0].message.content

        return result

    async def generate_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                             options: Dict[str, Any], dry_run: bool, api_key: str = None) -> str:
        messages, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return ''

        client = AsyncOpenAI(api_key=api_key)
        completion = await client.chat.completions.create(
            model=self.model_name, messages=messages,
            **options)

        LOGGER.info(f'Received response from {self.model_name}: "{completion.choices[0].message}".')

        result = completion.choices[0].message.content

        return result

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]],
                        options: Dict[str, Any],
                        dry_run: bool, api_key: str = None) -> Generator[str, None, None]:
        messages, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return

        client = OpenAI(api_key=api_key)
        completion_stream = client.chat.completions.create(model=self.model_name, messages=messages,
                                                           **options,
                                                           stream=True)

        result = ''

        for chunk in completion_stream:
            LOGGER.info(f'Received response (streaming) from {self.model_name}: "{chunk.choices[0].delta}".')

            delta = chunk.choices[0].delta.content
            if delta is not None:
                result += delta
                yield result

    async def generate_stream_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                                    options: Dict[str, Any],
                                    dry_run: bool, api_key: str = None) -> Generator[str, None, None]:
        messages, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return

        client = AsyncOpenAI(api_key=api_key)
        completion_stream = await client.chat.completions.create(model=self.model_name, messages=messages,
                                                                 **options,
                                                                 stream=True)

        result = ''

        async for chunk in completion_stream:
            LOGGER.info(f'Received response (streaming) from {self.model_name}: "{chunk.choices[0].delta}".')

            delta = chunk.choices[0].delta.content
            if delta is not None:
                result += delta
                yield result


class GPTImageGenerator(AbstractContentGenerator[str]):
    def __init__(self, model_name: str):
        super().__init__('OpenAI')
        self.model_name = model_name

    def _set_up(self, data: Dict[str, str | List[Dict[str, str | List]]],
                options: Dict[str, Any], dry_run: bool, api_key: str) -> Tuple:
        prompt = data.pop('prompt')
        if isinstance(prompt, list):
            raise RmlFormatException('Prompt must only contain string.')

        data: Dict[str, List[str]]
        update_options(options, data, GPT_IMAGE_OPTION_TYPES)

        LOGGER.info(f'Sending prompt to {self.model_name}: "{prompt}".')
        LOGGER.debug(f'Options: {options}.')

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')

        api_key = self.get_api_key(api_key)

        return prompt, options, api_key

    def generate(self, data: Dict[str, str | List[str]],
                 options: Dict[str, Any], dry_run: bool, api_key: str = None) -> str:
        prompt, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return ''

        client = OpenAI(api_key=api_key)
        image = client.images.generate(
            model=self.model_name, prompt=prompt,
            **options
        )

        LOGGER.info(f'Received response from {self.model_name}: "{image.data}".')

        result = image.data[0].url

        return result

    async def generate_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                             options: Dict[str, Any], dry_run: bool, api_key: str = None) -> str:
        prompt, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return ''

        client = AsyncOpenAI(api_key=api_key)
        image = await client.images.generate(
            model=self.model_name, prompt=prompt,
            **options
        )

        LOGGER.info(f'Received response from {self.model_name}: "{image.data}".')

        result = image.data[0].url

        return result

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]],
                        options: Dict[str, Any],
                        dry_run: bool, api_key: str = None) -> Generator[str, None, None]:
        raise NotImplementedError('Stream generation is not supported for image generation.')

    async def generate_stream_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                                    options: Dict[str, Any],
                                    dry_run: bool, api_key: str = None) -> Generator[str, None, None]:
        raise NotImplementedError('Stream generation is not supported for image generation.')
