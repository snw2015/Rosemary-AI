from typing import List, Dict, TypeAlias, Union

from .namespace import Namespace
from .transformer import RmlElement

LeafElement: TypeAlias = Union['RosemaryPetal', 'RosemaryTemplate']
RosemaryNamespace: TypeAlias = Namespace[LeafElement]


class RosemaryTemplate:
    def __init__(self, element: RmlElement,
                 variable_names: List[str], slot_vars: Dict[str, List[str]], namespace: RosemaryNamespace):
        self.element = element
        self.variable_names = variable_names
        self.slot_vars = slot_vars
        self.namespace = namespace


class RosemaryPetal:
    def __init__(self, formatter_rml: RmlElement, parser_rml: RmlElement, namespace: RosemaryNamespace,
                 variable_names: List[str], target: str = '', init: str = '{}'):
        self.formatter_rml = formatter_rml
        self.parser_rml = parser_rml
        self.namespace = namespace
        self.variable_names = variable_names
        self.target = target
        self.init = init

    def __str__(self):
        return f'Rosemary Petal'

    def __repr__(self):
        return self.__str__()
