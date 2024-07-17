from typing import Generator, Dict, Any, List

from ._option_types import CHAT_OPTION_TYPES, IMAGE_OPTION_TYPES
from ._utils import _shape_messages, _update_options
from .._utils._image import _image_to_data_uri  # noqa
from .generator import AbstractContentGenerator
from openai import OpenAI

from .._logger import LOGGER


class GPTChatGenerator(AbstractContentGenerator):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, data: Dict[str, str | List[Dict[str, str | List]]],
                 options: Dict[str, Any], dry_run: bool) -> str:
        messages = _shape_messages(data.pop('messages'))

        data: Dict[str, List[str]]
        _update_options(options, data, CHAT_OPTION_TYPES)

        LOGGER.info(f'Sending messages to {self.model_name}: "{messages}".')
        LOGGER.debug(f'Options: {options}.')

        api_key = options.pop('api_key', None)

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')
            return ''

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=self.model_name, messages=messages,
            **options)  # type: ignore

        LOGGER.info(f'Received response from {self.model_name}: "{completion.choices[0].message}".')

        result = completion.choices[0].message.content

        return result

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]],
                        options: Dict[str, Any], dry_run: bool) -> Generator[str, None, None]:
        messages = _shape_messages(data.pop('messages'))

        data: Dict[str, List[str]]
        _update_options(options, data, CHAT_OPTION_TYPES)

        LOGGER.info(f'Sending messages to {self.model_name}: "{messages}".')
        LOGGER.debug(f'Options: {options}.')

        api_key = options.pop('api_key', None)

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')
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


class GPTImageGenerator(AbstractContentGenerator):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, data: Dict[str, str | List[str]],
                 options: Dict[str, Any], dry_run: bool) -> str:
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
            return ''

        client = OpenAI(api_key=api_key)
        image = client.images.generate(
            model=self.model_name, prompt=prompt,
            **options
        )  # type: ignore

        LOGGER.info(f'Received response from {self.model_name}: "{image.data}".')

        result = image.data[0].url

        return result

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]],
                        options: Dict[str, Any], dry_run: bool) -> Generator[
            str, None, None]:
        raise NotImplementedError('Stream generation is not supported for image generation.')
