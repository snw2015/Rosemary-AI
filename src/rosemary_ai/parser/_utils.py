from .._logger import LOGGER
from .._utils._str_utils import did_you_mean  # noqa
from .transformer import RmlElement

RESERVED_ATTR_NAMES = {
    'import': {'path'},
    'template': {'name', 'param', 'var', 'slot'},
    'petal': {'name', 'param', 'var', 'target', 'model_name'},
    'formatter': set(),
    'parser': {'strict'},
    'img': {'src', 'src_eval'},
    'if': {'cond'},
    'for': {'range', 'in', 'var', 'slot', 'try'},
    'optional': {'required'},
    'list': set(),
    'list-item': set(),
    'dict': set(),
    'dict-item': {'key', 'key_eval'},
    'div': set(),
    'br': set(),
    'or': set(),
}


def _check_invalid_attributes(element: RmlElement, reserved_attr_names: set[str]):
    attribute_names = set(element.attributes.keys())
    invalid_attr_names = attribute_names - reserved_attr_names
    for name in invalid_attr_names:
        candidate = did_you_mean(name, reserved_attr_names)
        LOGGER.warning(f'Attribute "{name}" is not used in <{".".join(element.indicator)}>.' +
                       (f' Did you mean: "{candidate}"?' if candidate else ''))
