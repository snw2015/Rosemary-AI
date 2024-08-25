import inspect
from typing import Generator, Dict, Any, List, Tuple, Callable, TypeAlias

from openai import OpenAI, AsyncOpenAI
from openai.types import Moderation
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message_tool_call import Function

from ._utils import shape_messages, update_options
from .generator import AbstractContentGenerator
from .._logger import LOGGER
from ..exceptions import RmlFormatException, RequestFailedException
from ..multi_modal.image import Image

GptReturnType: TypeAlias = str | list[ChatCompletionMessageToolCall] | Dict[str, Any]


def _concatenate_delta() -> Generator[GptReturnType, ChoiceDelta, None]:
    text = None
    tool_calls = []

    while True:
        if text:
            delta = yield text  # type: ignore
        else:
            delta = yield tool_calls  # type: ignore
        delta: ChoiceDelta

        if delta.content:
            if not text:
                text = ''
            text += delta.content
        elif delta.tool_calls:
            for delta_tool_call in delta.tool_calls:
                tool_id = delta_tool_call.id
                arguments = delta_tool_call.function.arguments
                name = delta_tool_call.function.name

                if name is None and arguments is not None:  # function exists and arguments need to be updated
                    tool_call = tool_calls[-1]
                    tool_call.function.arguments += arguments
                elif name is not None:  # new function
                    tool_call = ChatCompletionMessageToolCall(
                        id=tool_id,
                        function=Function(
                            arguments='',
                            name=name,
                        ),
                        type='function'
                    )
                    tool_calls.append(tool_call)


class GPTChatGenerator(AbstractContentGenerator[GptReturnType]):
    def __init__(self, model_name: str):
        super().__init__('OpenAI')
        self.model_name = model_name

    def _set_up(self, data: Dict[str, str | List[Dict[str, str | List | Image]]],
                options: Dict[str, Any], dry_run: bool, api_key: str) -> Tuple:
        messages = shape_messages(data.pop('messages'))

        data: Dict[str, List[str]]
        update_options(options, data)

        if 'funcs' in options:
            funcs = options.pop('funcs')
            if not isinstance(funcs, list):
                raise RmlFormatException('"funcs" parameter must be a list of functions.')
            tools = []
            for func in funcs:
                if not isinstance(func, Callable):
                    raise RmlFormatException('Each function in "funcs" must be a callable.')
                else:
                    # Turn the function into description
                    params = inspect.signature(func).parameters
                    properties = {}
                    required = []
                    for param_name in params:
                        param = params[param_name]
                        if param.annotation == inspect.Parameter.empty or param.annotation == str:
                            param_type = 'string'
                        elif param.annotation == int:
                            param_type = 'integer'
                        elif param.annotation == float:
                            param_type = 'number'
                        elif param.annotation == bool:
                            param_type = 'boolean'
                        else:
                            raise RmlFormatException(
                                f'Unsupported parameter type: {param.annotation} for function {func.__name__}. '
                                f'For now, Rosemary only supports str, int, float, and bool.'
                            )
                        if param.default != inspect.Parameter.empty:
                            LOGGER.warning(f'Function {func.__name__} has default value for parameter {param_name}. '
                                           f'Rosemary does not support default value for now.')
                        properties[param_name] = {
                            'type': param_type
                        }
                        required.append(param_name)

                    tool = {
                        "type": "function",
                        "function": {
                            "name": func.__name__,
                            "description": func.__doc__,
                            "parameters": {
                                "type": "object",
                                "properties": properties,
                                "required": required,
                                "additionalProperties": False,
                            }
                        }
                    }
                    tools.append(tool)
            options['tools'] = tools

        return_json = False

        if 'return_json' in options:
            return_json = options.pop('return_json')

        if return_json:
            LOGGER.info(f'The "return_json" option is enabled. The messages and options sent to API will '
                        f'be directly returned in JSON format.')
        else:
            LOGGER.info(f'Sending messages to {self.model_name}: "{messages}".')
            LOGGER.debug(f'Options: {options}.')

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')

        api_key = self.get_api_key(api_key)

        return messages, options, api_key, return_json

    def _get_json_request(self, messages: List[Dict[str, str | List]], options: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'messages': messages,
            'model': self.model_name,
            **options
        }

    def _get_result_from_completion(self, completion: ChatCompletion) -> GptReturnType:
        choice = completion.choices[0]

        LOGGER.info(f'Received response from {self.model_name}: "{choice.message}".')

        if choice.finish_reason == 'tool_calls':
            return choice.message.tool_calls
        elif choice.finish_reason == 'stop':
            return choice.message.content
        else:
            raise RequestFailedException(f'Unexpected finish reason: {choice.finish_reason}.')

    def generate(self, data: Dict[str, str | List[Dict[str, str | List]]],
                 options: Dict[str, Any], dry_run: bool, api_key: str = None) -> GptReturnType:
        messages, options, api_key, return_json = self._set_up(data, options, dry_run, api_key)
        if dry_run:
            return ''

        if return_json:
            return self._get_json_request(messages, options)

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=self.model_name, messages=messages,
            **options)

        return self._get_result_from_completion(completion)

    async def generate_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                             options: Dict[str, Any], dry_run: bool, api_key: str = None) -> GptReturnType:
        messages, options, api_key, return_json = self._set_up(data, options, dry_run, api_key)
        if dry_run:
            return ''

        if return_json:
            return self._get_json_request(messages, options)

        client = AsyncOpenAI(api_key=api_key)
        completion = await client.chat.completions.create(
            model=self.model_name, messages=messages,
            **options)

        return self._get_result_from_completion(completion)

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]],
                        options: Dict[str, Any],
                        dry_run: bool, api_key: str = None) -> Generator[GptReturnType, None, None]:
        messages, options, api_key, return_json = self._set_up(data, options, dry_run, api_key)
        if dry_run:
            return

        if return_json:
            raise NotImplementedError('Stream mode is not supported for JSON return.')

        client = OpenAI(api_key=api_key)
        completion_stream = client.chat.completions.create(model=self.model_name, messages=messages,
                                                           **options,
                                                           stream=True)

        delta_stream = _concatenate_delta()
        next(delta_stream)

        for chunk in completion_stream:
            delta = chunk.choices[0].delta
            LOGGER.info(f'Received response (streaming) from {self.model_name}: "{delta}".')

            if chunk.choices[0].finish_reason is None:
                yield delta_stream.send(delta)

    async def generate_stream_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                                    options: Dict[str, Any],
                                    dry_run: bool, api_key: str = None) -> Generator[GptReturnType, None, None]:
        messages, options, api_key, return_json = self._set_up(data, options, dry_run, api_key)
        if dry_run:
            return

        if return_json:
            raise NotImplementedError('Stream mode is not supported for JSON return.')

        client = AsyncOpenAI(api_key=api_key)
        completion_stream = await client.chat.completions.create(model=self.model_name, messages=messages,
                                                                 **options,
                                                                 stream=True)

        delta_stream = _concatenate_delta()
        next(delta_stream)

        async for chunk in completion_stream:
            LOGGER.info(f'Received response (streaming) from {self.model_name}: "{chunk.choices[0].delta}".')

            delta = chunk.choices[0].delta
            if delta is not None:
                yield delta_stream.send(delta)


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

        for data in response.iter_bytes():
            yield data  # For TTS, when directly return the byte stream for playing

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

        async for data in await response.aiter_bytes():
            yield data  # For TTS, when directly return the byte stream for playing


class GPTModerationGenerator(AbstractContentGenerator[Moderation | None]):
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
                 options: Dict[str, Any], dry_run: bool, api_key: str = None) -> Moderation | None:
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
                             options: Dict[str, Any], dry_run: bool, api_key: str = None) -> Moderation | None:
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
