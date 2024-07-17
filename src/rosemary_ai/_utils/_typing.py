import inspect
from typing import Callable


def add_parameter_to_func(f: Callable, param_name: str, param_type):
    signature = inspect.signature(f)
    param = inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=param_type, default=None)
    parameters = list(signature.parameters.values()) + [param]
    new_signature = signature.replace(parameters=parameters)

    f.__signature__ = new_signature
