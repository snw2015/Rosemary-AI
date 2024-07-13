import functools
import inspect
from typing import Dict, Any

from .rosemary import get_function, get_function_stream


def petal(rosemary_name: str, function_name: str, stream=False,
          model_name: str = None, options: Dict[str, Any] = None):
    def decorator(func):
        signatures = inspect.signature(func)

        if stream:
            rosemary_function = get_function_stream(rosemary_name, function_name, signatures,
                                                    model_name, options)
        else:
            rosemary_function = get_function(rosemary_name, function_name, signatures,
                                             model_name, options)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return rosemary_function(*args, **kwargs)

        return wrapper

    return decorator
