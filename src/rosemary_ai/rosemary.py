from typing import Callable, Dict, Any, Tuple, Generator

from .models.generator_registry import get_generator
from .parser.executor import FormatExecutor, ParseExecutor
from .parser.leaf_elements import RosemaryPetal, _build_environment  # noqa
from .parser.traverse import traverse_all
from .parser.namespace import Namespace
from .parser.parser import RosemaryParser
from ._utils._str_utils import full_name_to_indicator  # noqa


def _generate(petal: RosemaryPetal, model_name: str, options: Dict[str, Any],
              target_obj, args: Dict[str, Any]) -> Any:
    if options is None:
        options = {}

    data = _format(petal, args)

    generator = get_generator(model_name)
    # TODO if no model name given, use default model defined in petal

    raw_str = generator.generate(data, options)

    target_obj, succeed = _parse(petal, args, raw_str, target_obj=target_obj)

    if succeed:
        return target_obj
    else:
        raise ValueError(f'Failed to parse {raw_str}')


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
        raise ValueError(f'Failed to parse {raw_str}')


class Rosemary:
    def __init__(self, src_path: str):
        self._build_from_src(src_path)

    def _build_from_src(self, src_path: str):
        rosemary_parser = RosemaryParser(src_path)
        self.namespace: Namespace = rosemary_parser.namespace

    def get_function(self, function_name: str) -> Callable:
        petal = self.namespace.get_by_indicator(full_name_to_indicator(function_name))

        def func(target_obj=None, model_name: str = None, options: Dict[str, Any] = None, **args) -> Any:
            return _generate(petal, model_name, options, target_obj, args)

        return func

    def get_function_bind(self, function_name: str) -> Callable:
        petal = self.namespace.get_by_indicator(full_name_to_indicator(function_name))

        def func(self_, target_obj=None, model_name: str = None, options: Dict[str, Any] = None, **args) -> Any:
            args['self'] = self_
            if target_obj is None:
                target_obj = self_

            return _generate(petal, model_name, options, target_obj, args)

        return func

    def get_function_stream(self, function_name: str) -> Callable:
        petal = self.namespace.get_by_indicator(full_name_to_indicator(function_name))

        def func(target_obj=None, model_name: str = None, options: Dict[str, Any] = None, **args) -> (
                Generator[Any, None, None]):
            for data in _generate_stream(petal, model_name, options, target_obj, args):
                yield data

        return func

    def get_function_stream_bind(self, function_name: str) -> Callable:
        petal = self.namespace.get_by_indicator(full_name_to_indicator(function_name))

        def func(self_, target_obj=None, model_name: str = None, options: Dict[str, Any] = None, **args) -> (
                Generator[Any, None, None]):
            args['self'] = self_
            if target_obj is None:
                target_obj = self_

            for data in _generate_stream(petal, model_name, options, target_obj, args):
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


def build(src_path: str) -> Rosemary:
    return Rosemary(src_path)


def _format(petal: RosemaryPetal, data: Dict[str, Any]) -> Any:
    if petal.formatter_rml is None:
        return data

    if petal.variable_names:
        data = {name: None for name in petal.variable_names} | data

    env = _build_environment(petal, data.copy())
    executor = FormatExecutor()

    succeed = traverse_all([env], petal.formatter_rml.children, executor)

    if not succeed:
        raise ValueError('Failed to format')

    return executor.get_result()


def _parse(petal: RosemaryPetal, data: Dict[str, Any], raw_str: str, target_obj=None) -> Tuple[Any, bool]:
    if petal.parser_rml is None:
        return raw_str, True

    if petal.variable_names:
        data = {name: None for name in petal.variable_names} | data

    if petal.target:
        data = data | {petal.target: target_obj if target_obj is not None else eval(petal.init)}

    env = _build_environment(petal, data)
    executor = ParseExecutor(raw_str)

    succeed = traverse_all([env], petal.parser_rml.children, executor)

    executor.activate_assignments(succeed)

    return data.get(petal.target), succeed
