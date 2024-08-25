from typing import List, Dict, Any, Callable

import httpx
import requests

from ..exceptions import RmlFormatException, RequestFailedException
from .._utils.image import image_to_data_uri
from ..multi_modal.image import Image


def shape_messages(data: List[Dict[str, str | List[str | Image]]],
                   text_formatter: Callable[[str], Any] = None,
                   image_formatter: Callable[[Image], Any] = None
                   ) -> List[Dict[str, str]]:
    messages = []

    for d in data:
        for role, content in d.items():
            if isinstance(content, str):
                messages.append({'role': role, 'content': content})
            if isinstance(content, Image):
                content_arr = _create_multimodal_arr([content], text_formatter, image_formatter)
                messages.append({'role': role, 'content': content_arr})
            if isinstance(content, List):
                content_arr = _create_multimodal_arr(content, text_formatter, image_formatter)
                messages.append({'role': role, 'content': content_arr})

    return messages


def _create_multimodal_arr(content: List[str | Image],
                           text_formatter: Callable[[str], Any] = None,
                           image_formatter: Callable[[Image], Any] = None
                           ) -> List[Dict[str, str]]:
    content_arr = []
    for c in content:
        if isinstance(c, str):
            if text_formatter is None:
                content_arr.append({'type': 'text', 'text': c})
            else:
                content_arr.append(text_formatter(c))
        elif isinstance(c, Image):
            c: Image
            if image_formatter is None:
                url = image_to_data_uri(c)
                content_arr.append({'type': 'image_url', 'image_url':
                    {'url': url}})
            else:
                content_arr.append(image_formatter(c))
    return content_arr


def update_options(options: Dict[str, Any], new_options: Dict[str, List[str]], option_types: Dict[str, Any] = None):
    """
    Update options with new options. The new options is raw data from formatter,
    so a list of string should be converted to a string.
    It will Also cast the values to the correct type.

    Legacy: Now we assume that the new options are already formatted.
    """

    # if option_types is None:
    #     option_types = {}

    # for key, value in new_options.items():
    # if key not in options:
    # if key in option_types:
    #     value = option_types[key](value)
    # options[key] = value

    options.update(new_options)


def _system_prompt_in_messages(messages):
    return any(message['role'] == 'system' for message in messages)


def reform_system_message(messages, provider: str):
    if messages[0]['role'] == 'system':
        system = messages[0]['content']
        messages = messages[1:]
    else:
        system = None
    if _system_prompt_in_messages(messages):
        raise NotImplementedError(f'Only the first message can be a system prompt in {provider}.')

    if not messages:
        raise NotImplementedError(f'At least one message is required in {provider}.')

    return messages, system


def check_response_status(response: requests.Response | httpx.Response) -> None:
    if response.status_code != 200:
        raise RequestFailedException(response)
