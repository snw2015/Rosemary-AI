from typing import Generator, Dict, Any, List

from .._utils._image import _image_to_data_uri  # noqa
from .generator import AbstractContentGenerator
from openai import OpenAI, NOT_GIVEN

from ..multi_modal.image import Image


def _shape_messages(data: List[Dict[str, str | List[str | Image]]]) -> List[Dict[str, str]]:
    messages = []

    for d in data:
        for role, content in d.items():
            if isinstance(content, str):
                messages.append({'role': role, 'content': content})
            if isinstance(content, List):
                content_arr = _glue_str(content)
                if len(content_arr) == 1 and content_arr[0]['type'] == 'text':
                    messages.append({'role': role, 'content': content_arr[0]['text']})
                else:
                    messages.append({'role': role, 'content': content_arr})

    return messages


def _glue_str(content: List[str | Image]) -> List[Dict[str, str]]:
    content_arr = []
    for c in content:
        if isinstance(c, str):
            if content_arr and content_arr[-1]['type'] == 'text':
                content_arr[-1]['text'] += c
            else:
                content_arr.append({'type': 'text', 'text': c})
        elif isinstance(c, Image):
            c: Image
            url = _image_to_data_uri(c)
            content_arr.append({'type': 'image_url', 'image_url':
                {'url': url}})
    return content_arr


class GPTChatGenerator(AbstractContentGenerator):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, data: Dict[str, str | List[Dict[str, str | List]]], options: Dict[str, Any]) -> str:
        messages = _shape_messages(data['messages'])

        api_key = options.get('api_key')
        temperature = options.get('temperature', NOT_GIVEN)
        max_tokens = options.get('max_tokens', NOT_GIVEN)

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=self.model_name, messages=messages,
            temperature=temperature, max_tokens=max_tokens)  # type: ignore

        response = completion.choices[0].message.content

        return response

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]], options: Dict[str, Any]) -> Generator[
            str, None, None]:
        messages = _shape_messages(data['messages'])

        api_key = options.get('api_key')
        temperature = options.get('temperature', NOT_GIVEN)
        max_tokens = options.get('max_tokens', NOT_GIVEN)

        client = OpenAI(api_key=api_key)
        completion_stream = client.chat.completions.create(model=self.model_name, messages=messages,
                                                           temperature=temperature, max_tokens=max_tokens,
                                                           stream=True)  # type: ignore

        response = ''

        for chunk in completion_stream:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                response += delta
                yield response


class GPTImageGenerator(AbstractContentGenerator):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, data: Dict[str, str | List[str]], options: Dict[str, Any]) -> str:
        prompt = data['prompt']
        if isinstance(prompt, list):
            prompt = ' '.join(prompt)

        api_key = options.get('api_key')
        quality = options.get('quality', NOT_GIVEN)
        size = options.get('size', NOT_GIVEN)
        style = options.get('style', NOT_GIVEN)

        client = OpenAI(api_key=api_key)
        image = client.images.generate(
            model=self.model_name, prompt=prompt,
            quality=quality, size=size, style=style)  # type: ignore

        response = image.data[0].url

        return response

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]], options: Dict[str, Any]) -> Generator[
            str, None, None]:
        raise NotImplementedError
