from typing import Generator, Dict, Any, List

from .._utils._image import _image_to_data_uri  # noqa
from .generator import AbstractContentGenerator
from openai import OpenAI

from ..multi_modal.image import Image


def _shape_messages(data: list[dict[str, str | List]]) -> list[dict[str, str]]:
    messages = []

    for d in data:
        for role, content in d.items():
            if isinstance(content, str):
                messages.append({'role': role, 'content': content})
            if isinstance(content, List):
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
                messages.append({'role': role, 'content': content_arr})

    return messages


class GPTGenerator(AbstractContentGenerator):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, data: list[dict[str, str]], options: Dict[str, Any]) -> str:
        messages = _shape_messages(data)
        print(messages)

        api_key = options.get('api_key')

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(model=self.model_name, messages=messages)  # type: ignore

        response = completion.choices[0].message.content
        print(response)

        return response

    def generate_stream(self, data: list[dict[str, str]], options: Dict[str, Any]) -> Generator[str, None, None]:
        messages = _shape_messages(data)

        api_key = options.get('api_key')

        client = OpenAI(api_key=api_key)
        completion_stream = client.chat.completions.create(model=self.model_name, messages=messages,  # type: ignore
                                                           stream=True)  # type: ignore

        response = ''

        for chunk in completion_stream:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                response += delta
                yield response
