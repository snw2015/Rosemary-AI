from typing import Generator, Dict, Any, List, Tuple

from ._option_types import CHAT_OPTION_TYPES, IMAGE_OPTION_TYPES
from ._utils import _shape_messages, _update_options
from .._utils._image import _image_to_data_uri  # noqa
from .generator import AbstractContentGenerator
from openai import OpenAI, AsyncOpenAI

from .._logger import LOGGER


class GPTChatGenerator(AbstractContentGenerator[str]):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _set_up(self, data: Dict[str, str | List[Dict[str, str | List]]],
                options: Dict[str, Any], dry_run: bool) -> Tuple:
        messages = _shape_messages(data.pop('messages'))

        data: Dict[str, List[str]]
        _update_options(options, data, CHAT_OPTION_TYPES)

        LOGGER.info(f'Sending messages to {self.model_name}: "{messages}".')
        LOGGER.debug(f'Options: {options}.')

        api_key = options.pop('api_key', None)

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')

        return messages, options, api_key

    def generate(self, data: Dict[str, str | List[Dict[str, str | List]]],
                 options: Dict[str, Any], dry_run: bool) -> str:
        messages, options, api_key = self._set_up(data, options, dry_run)

        if dry_run:
            return ''

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=self.model_name, messages=messages,
            **options)  # type: ignore

        LOGGER.info(f'Received response from {self.model_name}: "{completion.choices[0].message}".')

        result = completion.choices[0].message.content

        return result

    async def generate_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                             options: Dict[str, Any], dry_run: bool) -> str:
        messages, options, api_key = self._set_up(data, options, dry_run)

        if dry_run:
            return ''

        client = AsyncOpenAI(api_key=api_key)
        completion = await client.chat.completions.create(
            model=self.model_name, messages=messages,
            **options)  # type: ignore

        LOGGER.info(f'Received response from {self.model_name}: "{completion.choices[0].message}".')

        result = completion.choices[0].message.content

        return result

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]],
                        options: Dict[str, Any], dry_run: bool) -> Generator[str, None, None]:
        messages, options, api_key = self._set_up(data, options, dry_run)

        if dry_run:
            return

        client = OpenAI(api_key=api_key)
        completion_stream = client.chat.completions.create(model=self.model_name, messages=messages,
                                                           **options,
                                                           stream=True)  # type: ignore

        result = ''

        for chunk in completion_stream:
            LOGGER.info(f'Received response (streaming) from {self.model_name}: "{chunk.choices[0].delta}".')

            delta = chunk.choices[0].delta.content
            if delta is not None:
                result += delta
                yield result

    async def generate_stream_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                                    options: Dict[str, Any], dry_run: bool) -> Generator[str, None, None]:
        messages, options, api_key = self._set_up(data, options, dry_run)

        if dry_run:
            return

        client = AsyncOpenAI(api_key=api_key)
        completion_stream = await client.chat.completions.create(model=self.model_name, messages=messages,
                                                                 **options,
                                                                 stream=True)  # type: ignore

        result = ''

        async for chunk in completion_stream:
            LOGGER.info(f'Received response (streaming) from {self.model_name}: "{chunk.choices[0].delta}".')

            delta = chunk.choices[0].delta.content
            if delta is not None:
                result += delta
                yield result


class GPTImageGenerator(AbstractContentGenerator[str]):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _set_up(self, data: Dict[str, str | List[Dict[str, str | List]]],
                options: Dict[str, Any], dry_run: bool) -> Tuple:
        prompt = data.pop('prompt')
        if isinstance(prompt, list):
            prompt = ''.join(prompt)

        data: Dict[str, List[str]]
        _update_options(options, data, IMAGE_OPTION_TYPES)

        LOGGER.info(f'Sending prompt to {self.model_name}: "{prompt}".')
        LOGGER.debug(f'Options: {options}.')

        api_key = options.pop('api_key', None)

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')

        return prompt, options, api_key

    def generate(self, data: Dict[str, str | List[str]],
                 options: Dict[str, Any], dry_run: bool) -> str:
        prompt, options, api_key = self._set_up(data, options, dry_run)

        if dry_run:
            return ''

        client = OpenAI(api_key=api_key)
        image = client.images.generate(
            model=self.model_name, prompt=prompt,
            **options
        )  # type: ignore

        LOGGER.info(f'Received response from {self.model_name}: "{image.data}".')

        result = image.data[0].url

        return result

    async def generate_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                             options: Dict[str, Any], dry_run: bool) -> str:
        prompt, options, api_key = self._set_up(data, options, dry_run)

        if dry_run:
            return ''

        client = AsyncOpenAI(api_key=api_key)
        image = await client.images.generate(
            model=self.model_name, prompt=prompt,
            **options
        )  # type: ignore

        LOGGER.info(f'Received response from {self.model_name}: "{image.data}".')

        result = image.data[0].url

        return result

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]],
                        options: Dict[str, Any], dry_run: bool) -> Generator[str, None, None]:
        raise NotImplementedError('Stream generation is not supported for image generation.')

    async def generate_stream_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                                    options: Dict[str, Any], dry_run: bool) -> Generator[str, None, None]:
        raise NotImplementedError('Stream generation is not supported for image generation.')
