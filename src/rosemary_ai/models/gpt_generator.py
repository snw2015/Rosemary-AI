from typing import Generator, Dict, Any, List

from ._utils import _shape_messages
from .._utils._image import _image_to_data_uri  # noqa
from .generator import AbstractContentGenerator
from openai import OpenAI, NOT_GIVEN

from .._logger import LOGGER


class GPTChatGenerator(AbstractContentGenerator):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, data: Dict[str, str | List[Dict[str, str | List]]], options: Dict[str, Any]) -> str:
        messages = _shape_messages(data['messages'])

        LOGGER.info(f'Sending messages to {self.model_name}: "{messages}".')

        api_key = options.get('api_key')
        temperature = options.get('temperature', NOT_GIVEN)
        max_tokens = options.get('max_tokens', NOT_GIVEN)

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=self.model_name, messages=messages,
            temperature=temperature, max_tokens=max_tokens)  # type: ignore

        LOGGER.info(f'Received response from {self.model_name}: "{completion.choices[0].message}".')

        result = completion.choices[0].message.content

        return result

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]], options: Dict[str, Any]) -> Generator[
            str, None, None]:
        messages = _shape_messages(data['messages'])

        LOGGER.info(f'Sending messages to {self.model_name}: "{messages}".')

        api_key = options.get('api_key')
        temperature = options.get('temperature', NOT_GIVEN)
        max_tokens = options.get('max_tokens', NOT_GIVEN)

        client = OpenAI(api_key=api_key)
        completion_stream = client.chat.completions.create(model=self.model_name, messages=messages,
                                                           temperature=temperature, max_tokens=max_tokens,
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

    def generate(self, data: Dict[str, str | List[str]], options: Dict[str, Any]) -> str:
        prompt = data['prompt']
        if isinstance(prompt, list):
            prompt = ' '.join(prompt)

        LOGGER.info(f'Sending prompt to {self.model_name}: "{prompt}".')

        api_key = options.get('api_key')
        quality = options.get('quality', NOT_GIVEN)
        size = options.get('size', NOT_GIVEN)
        style = options.get('style', NOT_GIVEN)

        client = OpenAI(api_key=api_key)
        image = client.images.generate(
            model=self.model_name, prompt=prompt,
            quality=quality, size=size, style=style)  # type: ignore

        LOGGER.info(f'Received response from {self.model_name}: "{image.data}".')

        result = image.data[0].url

        return result

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]], options: Dict[str, Any]) -> Generator[
            str, None, None]:
        raise NotImplementedError
