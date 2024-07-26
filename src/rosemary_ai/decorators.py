import functools
import inspect
from typing import Dict, Any, Callable

from ._utils.typing_utils import add_parameter_to_func
from .rosemary import get_function, get_function_stream


def _without_extra_kwargs(kwargs: Dict[str, Any], func: Callable) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in inspect.signature(func).parameters}


def petal(rosemary_name: str, function_name: str, stream=False,
          model_name: str = None, options: Dict[str, Any] = None, api_key: str = None):
    def decorator(func):
        signatures = inspect.signature(func)
        is_async = inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)

        if is_async:
            if stream:
                @functools.wraps(func)
                async def wrapper(*args, **kwargs):
                    kwargs_without_extra = _without_extra_kwargs(kwargs, func)

                    rosemary_function = get_function_stream(rosemary_name, function_name, signatures,
                                                            model_name, options,
                                                            func(*args, **kwargs_without_extra),
                                                            True, api_key)

                    async for result in rosemary_function(*args, **kwargs):
                        yield result
            else:
                @functools.wraps(func)
                async def wrapper(*args, **kwargs):
                    kwargs_without_extra = _without_extra_kwargs(kwargs, func)

                    rosemary_function = get_function(rosemary_name, function_name, signatures,
                                                     model_name, options,
                                                     await func(*args, **kwargs_without_extra),
                                                     True, api_key)

                    return await rosemary_function(*args, **kwargs)
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                kwargs_without_extra = _without_extra_kwargs(kwargs, func)

                if stream:
                    rosemary_function = get_function_stream(rosemary_name, function_name, signatures,
                                                            model_name, options,
                                                            func(*args, **kwargs_without_extra),
                                                            False, api_key)

                else:
                    rosemary_function = get_function(rosemary_name, function_name, signatures,
                                                     model_name, options,
                                                     func(*args, **kwargs_without_extra),
                                                     False, api_key)

                return rosemary_function(*args, **kwargs)

        add_parameter_to_func(wrapper, 'target_obj', Any)
        add_parameter_to_func(wrapper, 'model_name', str)
        add_parameter_to_func(wrapper, 'options', Dict[str, Any])
        add_parameter_to_func(wrapper, 'max_tries', int, 1)
        add_parameter_to_func(wrapper, 'dry_run', bool, False)
        add_parameter_to_func(wrapper, 'api_key', str, None)

        return wrapper

    return decorator
