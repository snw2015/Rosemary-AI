import typing
from inspect import Signature
from typing import Callable, Dict, Any, Tuple, Generator

from ._logger import LOGGER
from .exceptions import ParsingFailedException
from .models.generator_registry import get_generator
from .parser.executor import FormatExecutor, ParseExecutor
from .parser.leaf_elements import RosemaryPetal
from .parser.environment import build_environment
from .parser.traverse import traverse_all
from .parser.namespace import Namespace
from .parser.parser import RosemaryParser
from ._utils._str_utils import full_name_to_indicator  # noqa

_EMPTY = Signature.empty


def _generate(petal: RosemaryPetal, model_name: str, options: Dict[str, Any],
              target_obj, args: Dict[str, Any]) -> Any:
    if options is None:
        options = {}

    data = _format(petal, args)

    generator = get_generator(model_name)
    # TODO if no model name given, use default model defined in petal

    raw_data = generator.generate(data, options)

    target_obj, succeed = _parse(petal, args, raw_data, target_obj=target_obj)

    if succeed:
        return target_obj
    else:
        raise ParsingFailedException(f'Failed to parse from the model response: {raw_data}.')


def _generate_stream(petal: RosemaryPetal, model_name: str, options: Dict[str, Any],
                     target_obj, args: Dict[str, Any]) -> Generator[Any, None, None]:
    if options is None:
        options = {}

    data = _format(petal, args)

    generator = get_generator(model_name)
    # TODO if no model name given, use default model defined in petal

    succeed = False
    raw_str = ''
    for raw_str in generator.generate_stream(data, options):
        target_obj, succeed = _parse(petal, args, raw_str, target_obj=target_obj)

        yield target_obj

    if not succeed:
        raise ParsingFailedException(f'Failed to parse from the model response: {raw_str}')


def _print_unsupported_types_hint(signatures: Signature):
    for type_ in (*[param.annotation for param in signatures.parameters.values()], signatures.return_annotation):
        try:
            isinstance('?', type_)
        except TypeError:
            LOGGER.info(f'Type check of type "{type_}" is not supported yet.')


def _fill_args(kwargs: Dict[str, Any], signature: Signature, args: Tuple[Any]) -> Dict[str, Any]:
    kwargs = kwargs.copy()

    if signature:
        params = signature.parameters

        if params:
            for i, param in enumerate(params.values()):
                if param.name not in kwargs:
                    if i < len(args):
                        kwargs[param.name] = args[i]
                    elif param.default is not _EMPTY:
                        kwargs[param.name] = param.default
                    else:
                        LOGGER.warning(
                            f'Argument {param.name} without default value is not given. Will use None.')

                try:
                    if param.annotation is not _EMPTY and not isinstance(kwargs[param.name], param.annotation):
                        LOGGER.warning(f'Argument "{param.name}" has value "{kwargs[param.name]}",'
                                       f' which is not of type {param.annotation}.')
                except TypeError:
                    pass

    return kwargs


def _check_return_type(result, type_):
    try:
        if (type_ is not _EMPTY and
                not isinstance(result, type_)):
            LOGGER.warning(f'Generated result "{result}" is not of type {type_}.')
    except TypeError:
        pass


class Rosemary:
    def __init__(self, src_path: str):
        self._build_from_src(src_path)

    def _build_from_src(self, src_path: str):
        rosemary_parser = RosemaryParser(src_path)
        self.namespace: Namespace = rosemary_parser.namespace

    def get_function(self, function_name: str, signature: Signature = None,
                     model_name: str = None, options: Dict[str, Any] = None) -> Callable:
        petal = self.namespace.get_by_indicator(full_name_to_indicator(function_name))
        model_name_ = model_name
        options_ = options

        _print_unsupported_types_hint(signature)

        def func(*args, target_obj=None, model_name: str = model_name_, options=None,
                 **kwargs) -> Any:
            full_args = _fill_args(kwargs, signature, args)

            if options is None:
                options = options_

            result = _generate(petal, model_name, options, target_obj, full_args)

            _check_return_type(result, signature.return_annotation)

            return result

        return func

    def get_function_stream(self, function_name: str, signature: Signature = None,
                            model_name: str = None, options: Dict[str, Any] = None) -> Callable:
        petal = self.namespace.get_by_indicator(full_name_to_indicator(function_name))
        model_name_ = model_name
        options_ = options

        _print_unsupported_types_hint(signature)

        data_type = None
        if signature.return_annotation is not _EMPTY:
            annotation = signature.return_annotation

            if issubclass(typing.get_origin(annotation), typing.Generator):
                data_type = typing.get_args(annotation)[0]
            else:
                LOGGER.warning(f'Return type "{annotation}" of "{function_name}" is not a generator type.')

        def func(*args, target_obj=None, model_name: str = model_name_,
                 options=None, **kwargs) -> (Generator[Any, None, None]):
            full_args = _fill_args(kwargs, signature, args)

            if options is None:
                options = options_

            for data in _generate_stream(petal, model_name, options, target_obj, full_args):
                if data_type:
                    _check_return_type(data, data_type)

                yield data

        return func

    def get_formatter(self, function_name: str) -> Callable:
        petal = self.namespace.get_by_indicator(full_name_to_indicator(function_name))

        def formatter(**args):
            return _format(petal, args)

        return formatter

    def get_parser(self, function_name: str) -> Callable:
        petal = self.namespace.get_by_indicator(full_name_to_indicator(function_name))

        def parser(raw_str: str, target_obj=None, **args):
            return _parse(petal, args, raw_str, target_obj)

        return parser


def _build(src_path: str) -> Rosemary:
    return Rosemary(src_path)


_ROSEMARY_INSTANCE = {}


def load(name: str, src_path: str):
    if src_path not in _ROSEMARY_INSTANCE:
        _ROSEMARY_INSTANCE[name] = _build(src_path)


def get_function(name: str, function_name: str, signature: Signature = None,
                 model_name: str = None, options: Dict[str, Any] = None) -> Callable:
    return _ROSEMARY_INSTANCE[name].get_function(function_name, signature, model_name, options)


def get_function_stream(name: str, function_name: str, signature: Signature,
                        model_name: str = None, options: Dict[str, Any] = None) -> Callable:
    return _ROSEMARY_INSTANCE[name].get_function_stream(function_name, signature, model_name, options)


def _format(petal: RosemaryPetal, data: Dict[str, Any]) -> Any:
    if petal.formatter_rml is None:
        return data

    if petal.variable_names:
        data = {name: None for name in petal.variable_names} | data

    env = build_environment(petal, data.copy())
    executor = FormatExecutor()

    succeed = traverse_all(env, petal.formatter_rml.children, executor)

    if not succeed:
        raise ValueError('Failed to format')

    return executor.get_result()


def _parse(petal: RosemaryPetal, data: Dict[str, Any], raw_data: Any, target_obj=None) -> Tuple[Any, bool]:
    if petal.parser_rml is None:
        return raw_data, True

    if petal.variable_names:
        data = {name: None for name in petal.variable_names} | data

    if petal.target:
        data = data | {petal.target: target_obj if target_obj is not None else eval(petal.init)}

    env = build_environment(petal, data)
    executor = ParseExecutor(raw_data, petal.target, target_obj)

    succeed = traverse_all(env, petal.parser_rml.children, executor)

    return executor.activate_assignments(succeed), succeed
