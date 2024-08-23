from typing import Generator, Dict, Any, List, Tuple, BinaryIO
import io

from openai.types import Moderation

from ._utils import shape_messages, update_options
from ..exceptions import RmlFormatException
from ..multi_modal.image import Image
from .generator import AbstractContentGenerator
from openai import OpenAI, AsyncOpenAI

from .._logger import LOGGER


class GPTChatGenerator(AbstractContentGenerator[str]):
    def __init__(self, model_name: str):
        super().__init__('OpenAI')
        self.model_name = model_name

    def _set_up(self, data: Dict[str, str | List[Dict[str, str | List | Image]]],
                options: Dict[str, Any], dry_run: bool, api_key: str) -> Tuple:
        messages = shape_messages(data.pop('messages'))

        data: Dict[str, List[str]]
        update_options(options, data)

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
        update_options(options, data)

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


class GPTEmbeddingGenerator(AbstractContentGenerator[List[float]]):
    def __init__(self, model_name: str):
        super().__init__('OpenAI')
        self.model_name = model_name

    def _set_up(self, data: Dict[str, str | List[Dict[str, str | List]]],
                options: Dict[str, Any], dry_run: bool, api_key: str) -> Tuple:
        prompt = data.pop('input')
        if isinstance(prompt, list):
            raise RmlFormatException('Embedding input must only contain string.')

        data: Dict[str, List[str]]
        # update_options(options, data, EMBEDDING_OPTION_TYPES)
        update_options(options, data)

        LOGGER.info(f'Sending input to {self.model_name}: "{prompt}".')
        LOGGER.debug(f'Options: {options}.')

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')

        api_key = self.get_api_key(api_key)

        return prompt, options, api_key

    def generate(self, data: Dict[str, str | List[str]],
                 options: Dict[str, Any], dry_run: bool, api_key: str = None) -> List[float]:
        prompt, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return []

        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=prompt,
            model=self.model_name,
            **options
        )

        LOGGER.info(f'Received response from {self.model_name}: "{response.data[0]}".')

        result = response.data[0].embedding

        return result

    async def generate_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                             options: Dict[str, Any], dry_run: bool, api_key: str = None) -> List[float]:
        prompt, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return []

        client = AsyncOpenAI(api_key=api_key)
        response = await client.embeddings.create(
            input=prompt,
            model=self.model_name,
            **options
        )

        LOGGER.info(f'Received response from {self.model_name}: "{response.data[0]}".')

        result = response.data[0].embedding

        return result

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]],
                        options: Dict[str, Any],
                        dry_run: bool, api_key: str = None) -> Generator[str, None, None]:
        raise NotImplementedError('Stream mode is not supported for embeddings.')

    async def generate_stream_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                                    options: Dict[str, Any],
                                    dry_run: bool, api_key: str = None) -> Generator[str, None, None]:
        raise NotImplementedError('Stream mode is not supported for embeddings.')


class WhisperGenerator(AbstractContentGenerator[str]):
    def __init__(self, model_name: str):
        super().__init__('OpenAI')
        self.model_name = model_name

    def _set_up(self, data: Dict[str, Any],
                options: Dict[str, Any], dry_run: bool, api_key: str) -> Tuple:
        file_path = data.pop('file_path')
        if not isinstance(file_path, str):
            raise RmlFormatException('Whisper input must contain the audio file path as str.')

        data: Dict[str, List[str]]
        update_options(options, data)

        LOGGER.info(f'Sending input to {self.model_name}: "{file_path}".')
        LOGGER.debug(f'Options: {options}.')

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')

        api_key = self.get_api_key(api_key)

        return file_path, options, api_key

    def generate(self, data: Dict[str, Any],
                 options: Dict[str, Any], dry_run: bool, api_key: str = None) -> str:
        file_path, options, api_key = self._set_up(data, options, dry_run, api_key)

        with open(file_path, 'rb') as file:
            if dry_run:
                return ''

            client = OpenAI(api_key=api_key)
            response = client.audio.transcriptions.create(
                model=self.model_name,
                file=file,
                **options
            )

            LOGGER.info(f'Received response from {self.model_name}: "{response}".')

            result = response.text

            return result

    async def generate_async(self, data: Dict[str, Any],
                             options: Dict[str, Any], dry_run: bool, api_key: str = None) -> str:

        file_path, options, api_key = self._set_up(data, options, dry_run, api_key)

        with open(file_path, 'rb') as file:
            if dry_run:
                return ''

            client = AsyncOpenAI(api_key=api_key)
            response = await client.audio.transcriptions.create(
                model=self.model_name,
                file=file,
                **options
            )

            LOGGER.info(f'Received response from {self.model_name}: "{response}".')

            result = response.text

            return result

    def generate_stream(self, data: Dict[str, Any],
                        options: Dict[str, Any],
                        dry_run: bool, api_key: str = None) -> Generator[str, None, None]:
        raise NotImplementedError('Stream mode is not supported for Whisper.')

    async def generate_stream_async(self, data: Dict[str, Any],
                                    options: Dict[str, Any],
                                    dry_run: bool, api_key: str = None) -> Generator[str, None, None]:
        raise NotImplementedError('Stream mode is not supported for Whisper.')


class OpenAITTSGenerator(AbstractContentGenerator[bytes]):
    def __init__(self, model_name: str):
        super().__init__('OpenAI')
        self.model_name = model_name

    def _set_up(self, data: Dict[str, Any],
                options: Dict[str, Any], dry_run: bool, api_key: str) -> Tuple:
        text = data.pop('text')
        if not isinstance(text, str):
            raise RmlFormatException('TTS input must contain the text as str.')

        data: Dict[str, List[str]]
        update_options(options, data)

        LOGGER.info(f'Sending input to {self.model_name}: "{text}".')
        LOGGER.debug(f'Options: {options}.')

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')

        api_key = self.get_api_key(api_key)

        return text, options, api_key

    def generate(self, data: Dict[str, Any],
                 options: Dict[str, Any], dry_run: bool, api_key: str = None) -> bytes:
        text, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return b''

        client = OpenAI(api_key=api_key)
        response = client.audio.speech.create(
            model=self.model_name,
            input=text,
            **options
        )

        LOGGER.info(f'Received response from {self.model_name}: "{response}".')

        result = b''
        for data in response.iter_bytes():
            result += data

        return result

    async def generate_async(self, data: Dict[str, Any],
                             options: Dict[str, Any], dry_run: bool, api_key: str = None) -> bytes:
        text, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return b''

        client = AsyncOpenAI(api_key=api_key)
        response = await client.audio.speech.create(
            model=self.model_name,
            input=text,
            **options
        )

        LOGGER.info(f'Received response from {self.model_name}: "{response}".')

        result = b''

        async for data in await response.aiter_bytes():
            result += data

        return result

    def generate_stream(self, data: Dict[str, Any],
                        options: Dict[str, Any],
                        dry_run: bool, api_key: str = None) -> Generator[bytes, None, None]:
        text, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return

        client = OpenAI(api_key=api_key)
        response = client.audio.speech.create(
            model=self.model_name,
            input=text,
            **options
        )

        LOGGER.info(f'Received response from {self.model_name}: "{response}".')

        result = b''
        for data in response.iter_bytes():
            yield data

    async def generate_stream_async(self, data: Dict[str, Any],
                                    options: Dict[str, Any],
                                    dry_run: bool, api_key: str = None) -> Generator[bytes, None, None]:
        text, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return

        client = AsyncOpenAI(api_key=api_key)
        response = await client.audio.speech.create(
            model=self.model_name,
            input=text,
            **options
        )

        LOGGER.info(f'Received response from {self.model_name}: "{response}".')

        result = b''

        async for data in await response.aiter_bytes():
            yield data


class GPTModerationGenerator(AbstractContentGenerator[Moderation]):
    def __init__(self, model_name: str):
        super().__init__('OpenAI')
        self.model_name = model_name

    def _set_up(self, data: str,
                options: Dict[str, Any], dry_run: bool, api_key: str) -> Tuple:
        text = data

        LOGGER.info(f'Sending input to {self.model_name}: "{text}".')
        LOGGER.debug(f'Options: {options}.')

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')

        api_key = self.get_api_key(api_key)

        return text, options, api_key

    def generate(self, data: str,
                 options: Dict[str, Any], dry_run: bool, api_key: str = None) -> Moderation:
        text, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return None

        client = OpenAI(api_key=api_key)
        response = client.moderations.create(
            model=self.model_name,
            input=text,
        )

        LOGGER.info(f'Received response from {self.model_name}: "{response}".')

        result = response.results[0]

        return result

    async def generate_async(self, data: str,
                             options: Dict[str, Any], dry_run: bool, api_key: str = None) -> Moderation:
        text, options, api_key = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return None

        client = AsyncOpenAI(api_key=api_key)
        response = await client.moderations.create(
            model=self.model_name,
            input=text
        )

        LOGGER.info(f'Received response from {self.model_name}: "{response}".')

        result = response.results[0]

        return result

    def generate_stream(self, data: str,
                        options: Dict[str, Any],
                        dry_run: bool, api_key: str = None) -> Generator[Moderation, None, None]:
        raise NotImplementedError('Stream mode is not supported for moderation.')

    async def generate_stream_async(self, data: str,
                                    options: Dict[str, Any],
                                    dry_run: bool, api_key: str = None) -> Generator[Moderation, None, None]:
        raise NotImplementedError('Stream mode is not supported for moderation.')

