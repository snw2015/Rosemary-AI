from typing import List, Dict, Any

from .._utils._image import _image_to_data_uri  # noqa
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


def _update_options(options: Dict[str, Any], new_options: Dict[str, List[str]], option_types: Dict[str, Any]):
    for key, value_arr in new_options.items():
        value_arr: List[str]
        if key not in options:
            value = value_arr[0]
            if key in option_types:
                value = option_types[key](value)
            options[key] = value


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
        raise NotImplementedError(f'At least one message is required in Claude {provider}.')

    return messages, system