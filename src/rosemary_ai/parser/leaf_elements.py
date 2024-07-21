from typing import List, Dict, TypeAlias, Union

from .namespace import Namespace
from .transformer import RmlElement

LeafElement: TypeAlias = Union['RosemaryPetal', 'RosemaryTemplate']
RosemaryNamespace: TypeAlias = Namespace[LeafElement]


class RosemaryTemplate:
    def __init__(self, element: RmlElement,
                 parameter_names: List[str], slot_params: Dict[str, List[str]], namespace: RosemaryNamespace):
        self.element = element
        self.parameter_names = parameter_names
        self.slot_params = slot_params
        self.namespace = namespace


class RosemaryPetal:
    def __init__(self, formatter_rml: RmlElement, parser_rml: RmlElement, namespace: RosemaryNamespace,
                 parameter_names: List[str], target: str, init: str, is_parse_strict: bool,
                 default_model_name: str):
        self.formatter_rml = formatter_rml
        self.parser_rml = parser_rml
        self.namespace = namespace
        self.parameter_names = parameter_names
        self.target = target
        self.init = init
        self.is_parse_strict = is_parse_strict
        self.default_model_name = default_model_name

    def __str__(self):
        return f'Rosemary Petal'

    def __repr__(self):
        return self.__str__()
