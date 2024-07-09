import functools
from typing import Dict, Any

from .rosemary import get_function, get_function_stream, get_function_bind, get_function_stream_bind


def petal(rosemary_name: str, function_name: str, stream=False, bind=False,
          model_name: str = None, options: Dict[str, Any] = None):
    if stream:
        if bind:
            rosemary_function = get_function_stream_bind(rosemary_name, function_name, model_name, options)
        else:
            rosemary_function = get_function_stream(rosemary_name, function_name, model_name, options)
    else:
        if bind:
            rosemary_function = get_function_bind(rosemary_name, function_name, model_name, options)
        else:
            rosemary_function = get_function(rosemary_name, function_name, model_name, options)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return rosemary_function(*args, **kwargs)

        return wrapper

    return decorator
