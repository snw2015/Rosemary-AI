import typing
from inspect import Signature
from typing import Callable, Dict, Any, Tuple, Generator

from ._logger import LOGGER
from ._utils.dict_utils import options_with_default
from ._utils.typing_utils import isinstance_
from .exceptions import ParsingFailedException, RmlFormatException
from .models.generator_registry import get_generator
from .parser.executor import FormatExecutor, ParseExecutor
from .parser.leaf_elements import RosemaryPetal
from .parser.environment import build_environment
from .parser.traverse import traverse_all
from .parser.namespace import Namespace
from .parser.parser import RosemaryParser
from ._utils.str_utils import full_name_to_indicator  # noqa

_EMPTY = Signature.empty
_MAX_TRIES = 1000


def _generate(petal: RosemaryPetal, model_name: str, options: Dict[str, Any],
              dry_run: bool, dry_run_val,
              target_obj, args: Dict[str, Any],
              api_key: str) -> Any:
    if options is None:
        options = {}

    data = _format(petal, args)

    generator = get_generator(model_name if model_name else petal.default_model_name)

    raw_data = generator.generate(data, options, dry_run, api_key)
    if dry_run:
        raw_data = dry_run_val

    target_obj, succeed = _parse(petal, args, raw_data, target_obj)

    if succeed:
        return target_obj
    else:
        raise ParsingFailedException(f'Failed to parse from the model response: {raw_data}.')


async def _generate_async(petal: RosemaryPetal, model_name: str, options: Dict[str, Any],
                          dry_run: bool, dry_run_val,
                          target_obj, args: Dict[str, Any],
                          api_key: str) -> Any:
    if options is None:
        options = {}

    data = _format(petal, args)

    generator = get_generator(model_name if model_name else petal.default_model_name)

    raw_data = await generator.generate_async(data, options, dry_run, api_key)

    if dry_run:
        raw_data = dry_run_val

    target_obj, succeed = _parse(petal, args, raw_data, target_obj)

    if succeed:
        return target_obj
    else:
        raise ParsingFailedException(f'Failed to parse from the model response: {raw_data}.')


def _generate_stream(petal: RosemaryPetal, model_name: str, options: Dict[str, Any],
                     dry_run: bool, dry_run_generator: Generator,
                     target_obj, args: Dict[str, Any],
                     api_key: str) -> Generator[Any, None, None]:
    if options is None:
        options = {}

    data = _format(petal, args)

    generator = get_generator(model_name if model_name else petal.default_model_name)

    succeed = False
    raw_data = None

    if not dry_run:
        for raw_data in generator.generate_stream(data, options, dry_run, api_key):
            target_obj, succeed = _parse(petal, args, raw_data, target_obj)

            yield target_obj
    else:
        # For logging purpose
        for _ in generator.generate_stream(data, options, dry_run, api_key):
            pass

        for raw_data in dry_run_generator:
            target_obj, succeed = _parse(petal, args, raw_data, target_obj)

            yield target_obj

    if not succeed:
        raise ParsingFailedException(f'Failed to parse from the model response: {raw_data}')


async def _generate_stream_async(petal: RosemaryPetal, model_name: str, options: Dict[str, Any],
                                 dry_run: bool, dry_run_generator,
                                 target_obj, args: Dict[str, Any],
                                 api_key: str) -> Generator[Any, None, None]:
    if options is None:
        options = {}

    data = _format(petal, args)

    generator = get_generator(model_name if model_name else petal.default_model_name)

    succeed = False
    raw_data = None

    if not dry_run:
        async for raw_data in generator.generate_stream_async(data, options, dry_run, api_key):
            target_obj, succeed = _parse(petal, args, raw_data, target_obj)

            yield target_obj
    else:
        # For logging purpose
        async for _ in generator.generate_stream_async(data, options, dry_run, api_key):
            pass

        async for raw_data in dry_run_generator:
            target_obj, succeed = _parse(petal, args, raw_data, target_obj)

            yield target_obj

    if not succeed:
        raise ParsingFailedException(f'Failed to parse from the model response: {raw_data}')


def _print_unsupported_types_hint(signatures: Signature):
    for type_ in (*[param.annotation for param in signatures.parameters.values()], signatures.return_annotation):
        if type_ is _EMPTY:
            continue

        try:
            isinstance_('', type_)
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
                    if param.annotation is not _EMPTY and not isinstance_(kwargs[param.name], param.annotation):
                        LOGGER.warning(f'Argument "{param.name}" has value "{repr(kwargs[param.name])}",'
                                       f' which is not of type {param.annotation}.')
                except TypeError:
                    LOGGER.info(f'Type check of type "{param.annotation}" is not supported yet.')

    return kwargs


def _check_return_type(result, type_):
    try:
        if type_ is not _EMPTY and not isinstance_(result, type_):
            LOGGER.warning(f'Generated result "{repr(result)}" is not of type {type_}.')
    except TypeError:
        LOGGER.info(f'Type check of type "{type_}" is not supported yet.')


def _format(petal: RosemaryPetal, data: Dict[str, Any]) -> Any:
    if petal.formatter_rml is None:
        return data

    received_names = set(data.keys())
    expected_names = set(petal.parameter_names) | {'self'}

    names_only_in_data = received_names - expected_names
    for name in names_only_in_data:
        LOGGER.warning(f'Argument "{name}" is not used in the petal "{petal.name}", will be ignored.')

    names_only_in_petal = set(petal.parameter_names) - received_names
    for name in names_only_in_petal:
        LOGGER.warning(f'Argument "{name}" is not given, will be set to None.')

    data_with_default = {name: data[name] if name in data else None for name in expected_names}

    env = build_environment(petal, data_with_default)
    executor = FormatExecutor()

    succeed = traverse_all(env, petal.formatter_rml.children, executor)

    if not succeed:
        raise RmlFormatException('Failed to format')

    return executor.get_result()


def _parse(petal: RosemaryPetal, data: Dict[str, Any], raw_data: Any, target_obj=None) -> Tuple[Any, bool]:
    if petal.parser_rml is None:
        return raw_data, True

    if petal.parameter_names:
        data = {name: None for name in petal.parameter_names} | data

    if petal.target:
        if target_obj is None:
            assert petal.init is not None
            target_obj = eval(petal.init)
        data[petal.target] = target_obj

    env = build_environment(petal, data)
    executor = ParseExecutor(raw_data, petal.target, target_obj, petal.is_parse_strict)

    succeed = traverse_all(env, petal.parser_rml.children, executor)

    return executor.activate_assignments(succeed), succeed


class Rosemary:
    def __init__(self, src_path: str):
        self._build_from_src(src_path)

    def _build_from_src(self, src_path: str):
        rosemary_parser = RosemaryParser(src_path)
        self.namespace: Namespace = rosemary_parser.namespace

    def get_function(self, function_name: str, signature: Signature = None,
                     model_name: str = None, options: Dict[str, Any] = None,
                     dry_run_val=None, is_async: bool = False) -> Callable:
        petal = self.namespace.get_by_indicator(full_name_to_indicator(function_name))
        default_model_name = model_name
        default_options = options

        _print_unsupported_types_hint(signature)

        def __set_up(kwargs: Dict[str, Any], args: Tuple[Any], options_: Dict[str, Any], max_tries: int) -> Tuple:
            full_args = _fill_args(kwargs, signature, args)

            options_ = options_with_default(options_, default_options)

            inf_tries = False
            if max_tries < 0 or max_tries > _MAX_TRIES:
                max_tries = _MAX_TRIES
                inf_tries = True

            return full_args, options_, max_tries, inf_tries

        def __handle_exception(e: ParsingFailedException, time_try: int, max_tries: int, inf_tries: bool):
            LOGGER.info(str(e))
            if time_try < max_tries - 1:
                if inf_tries:
                    LOGGER.info(f'Retrying... ({time_try + 2})')
                elif max_tries > 1:
                    LOGGER.info(f'Retrying... ({time_try + 2}/{max_tries})')

        if is_async:
            async def func(*args, target_obj=None, model_name: str = default_model_name, options=None,
                           max_tries: int = 1, dry_run: bool = False, api_key: str = None,
                           **kwargs) -> Any:
                full_args, options_, max_tries, inf_tries = __set_up(kwargs, args, options, max_tries)

                for time_try in range(max_tries):
                    try:
                        result = await _generate_async(petal, model_name, options_, dry_run,
                                                       dry_run_val, target_obj, full_args, api_key)
                    except ParsingFailedException as e:
                        __handle_exception(e, time_try, max_tries, inf_tries)
                        continue

                    _check_return_type(result, signature.return_annotation)

                    return result

                raise ParsingFailedException(f'Failed to parse from the model response after {max_tries} tries.')

        else:

            def func(*args, target_obj=None, model_name: str = default_model_name, options=None,
                     max_tries: int = 1, dry_run: bool = False, api_key: str = None,
                     **kwargs) -> Any:
                full_args, options_, max_tries, inf_tries = __set_up(kwargs, args, options, max_tries)

                for time_try in range(max_tries):
                    try:
                        result = _generate(petal, model_name, options_, dry_run, dry_run_val,
                                           target_obj, full_args, api_key)
                    except ParsingFailedException as e:
                        __handle_exception(e, time_try, max_tries, inf_tries)
                        continue

                    _check_return_type(result, signature.return_annotation)

                    return result

                raise ParsingFailedException(f'Failed to parse from the model response after {max_tries} tries.')

        return func

    def get_function_stream(self, function_name: str, signature: Signature = None,
                            model_name: str = None, options: Dict[str, Any] = None,
                            dry_run_generator: Generator = None,
                            is_async: bool = False) -> Callable:
        petal = self.namespace.get_by_indicator(full_name_to_indicator(function_name))
        default_model_name = model_name
        default_options = options

        _print_unsupported_types_hint(signature)

        data_type = None
        if signature.return_annotation is not _EMPTY:
            annotation = signature.return_annotation

            if issubclass(typing.get_origin(annotation), typing.Generator):
                data_type = typing.get_args(annotation)[0]
            else:
                LOGGER.warning(f'Return type "{annotation}" of "{function_name}" is not a generator type.')

        def __set_up(kwargs: Dict[str, Any], args: Tuple[Any], options_: Dict[str, Any], max_tries: int) -> Tuple:
            full_args = _fill_args(kwargs, signature, args)

            options_ = options_with_default(options_, default_options)

            if max_tries != 1:
                LOGGER.warning('max_tries is not supported in stream mode. Will only try once.')

            return full_args, options_

        if is_async:
            async def func(*args, target_obj=None, model_name: str = default_model_name, options=None,
                           max_tries: int = 1, dry_run: bool = False, api_key: str = None,
                           **kwargs) -> (Generator[Any, None, None]):
                full_args, options_ = __set_up(kwargs, args, options, max_tries)

                async for data in _generate_stream_async(petal, model_name, options_,
                                                         dry_run, dry_run_generator,
                                                         target_obj, full_args, api_key):
                    if data_type:
                        _check_return_type(data, data_type)

                    yield data
        else:
            def func(*args, target_obj=None, model_name: str = default_model_name, options=None,
                     max_tries: int = 1, dry_run: bool = False, api_key: str = None,
                     **kwargs) -> (Generator[Any, None, None]):
                full_args, options_ = __set_up(kwargs, args, options, max_tries)

                for data in _generate_stream(petal, model_name, options_,
                                             dry_run, dry_run_generator,
                                             target_obj, full_args, api_key):
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
                 model_name: str = None, options: Dict[str, Any] = None, dry_run_val=None,
                 is_async: bool = False) -> Callable:
    return _ROSEMARY_INSTANCE[name].get_function(
        function_name, signature, model_name, options, dry_run_val, is_async
    )


def get_function_stream(name: str, function_name: str, signature: Signature,
                        model_name: str = None, options: Dict[str, Any] = None,
                        dry_run_generator: Generator = None,
                        is_async: bool = False) -> Callable:
    return _ROSEMARY_INSTANCE[name].get_function_stream(function_name, signature, model_name,
                                                        options, dry_run_generator, is_async)
