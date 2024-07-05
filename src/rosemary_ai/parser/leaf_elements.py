from typing import List, Dict, Any, TypeAlias, Union

from .namespace import Namespace
from .transformer import RmlElement

LeafElement: TypeAlias = Union['RosemaryPetal', 'RosemaryTemplate']
RosemaryNamespace: TypeAlias = Namespace[LeafElement]
VariableContext: TypeAlias = Dict[str, Any]


class Environment:
    def __init__(self, context: VariableContext, slots: Dict, namespace: RosemaryNamespace):
        self.context = context
        self.slots = slots
        self.namespace = namespace

    def __copy__(self):
        return Environment(self.context.copy(), self.slots.copy(), self.namespace)


class RosemaryTemplate:
    def __init__(self, element: RmlElement,
                 variable_names: List[str], slot_names: List[str], namespace: RosemaryNamespace):
        self.element = element
        self.variable_names = variable_names
        self.slot_names = slot_names
        self.namespace = namespace


class RosemaryPetal:
    def __init__(self, formatter_rml: RmlElement, parser_rml: RmlElement, namespace: RosemaryNamespace,
                 target: str, init: str = '{}'):
        self.formatter_rml = formatter_rml
        self.parser_rml = parser_rml
        self.namespace = namespace
        self.target = target
        self.init = init

    def __str__(self):
        return f'Rosemary Petal'

    def __repr__(self):
        return self.__str__()


def _build_environment(petal, data: VariableContext) -> Environment:
    context = data
    slots = {}
    namespace = petal.namespace
    return Environment(context, slots, namespace)


def rml_to_petal(tree: RmlElement, namespace: RosemaryNamespace) -> RosemaryPetal:
    formatter = None
    parser = None
    for child in tree.children:
        if child.is_text:
            continue
        elif child.indicator == ('formatter',):
            formatter = child
        elif child.indicator == ('parser',):
            parser = child
        else:
            raise ValueError(f'Unknown element {child.indicator}')
    target = None
    if 'target' in tree.attributes:
        target = tree.attributes['target']
    return RosemaryPetal(formatter, parser, namespace, target,
                         tree.attributes['init'] if 'init' in tree.attributes else '{}')


def rml_to_template(tree: RmlElement, namespace: RosemaryNamespace) -> RosemaryTemplate:
    variables = []
    if 'var' in tree.attributes:
        variables = list(map(str.strip, tree.attributes['var'].split(',')))

    slots = []
    if 'slot' in tree.attributes:
        slots = list(map(str.strip, tree.attributes['slot'].split(',')))

    return RosemaryTemplate(tree, variables, slots, namespace)
