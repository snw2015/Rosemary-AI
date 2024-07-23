import inspect
import typing
from typing import Callable


def add_parameter_to_func(f: Callable, param_name: str, param_type, default_value=None):
    signature = inspect.signature(f)
    param = inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              annotation=param_type, default=default_value)
    parameters = list(signature.parameters.values()) + [param]
    new_signature = signature.replace(parameters=parameters)

    f.__signature__ = new_signature


def isinstance_(obj, type_):
    if isinstance(type_, str):
        return obj.__class__.__name__ == type_
    elif typing.get_origin(type_) is None:
        return isinstance(obj, type_)
    else:
        return isinstance(obj, typing.get_origin(type_))
