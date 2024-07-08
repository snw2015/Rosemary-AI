from typing import List, Dict, TypeAlias, Union, Tuple, Set

from .data_expression import DataExpression, VariableContext
from .namespace import Namespace
from .transformer import RmlElement

LeafElement: TypeAlias = Union['RosemaryPetal', 'RosemaryTemplate']
RosemaryNamespace: TypeAlias = Namespace[LeafElement]


class Slot:
    def __init__(self, elements_with_info: List[Tuple[RmlElement, 'Environment', VariableContext]],
                 variable_names: List[str], is_inf: bool = False):
        self.element_with_info = elements_with_info
        self.variable_names = variable_names
        self.is_inf = is_inf

    def append(self, element: RmlElement, environment: 'Environment', var_context: VariableContext):
        self.element_with_info.append((element, environment, var_context))

    def pop(self) -> Tuple[RmlElement, 'Environment', VariableContext]:
        if self.is_inf and len(self.element_with_info) == 1:
            return self.element_with_info[-1]
        else:
            return self.element_with_info.pop()

    def reverse(self):
        self.element_with_info.reverse()

    def has_next(self) -> bool:
        return self.is_inf or bool(self.element_with_info)


Slots: TypeAlias = Dict[str, Slot]


class Environment:
    def __init__(self, context: VariableContext, slots: Slots, namespace: RosemaryNamespace):
        self.context = context
        self.slots = slots
        self.namespace = namespace

    def __copy__(self):
        return Environment(self.context.copy(), self.slots.copy(), self.namespace)

    def eval(self, expr: str | DataExpression, need_copy=True):
        if isinstance(expr, str):
            expr = DataExpression(expr)
        return expr.evaluate(self.context, need_copy)

    def exec(self, expr: str, need_copy=True):
        if isinstance(expr, str):
            expr = DataExpression(expr)
        expr.execute(self.context, need_copy)

    def __str__(self):
        return f'Environment<{self.context} | {self.slots}>'


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
    variable_names = []
    if 'var' in tree.attributes:
        variable_names = list(map(str.strip, tree.attributes['var'].split(',')))
    return RosemaryPetal(formatter, parser, namespace, variable_names, target,
                         tree.attributes['init'] if 'init' in tree.attributes else '{}')


def _get_slot_var(str_repr: str) -> Dict[str, List[str]]:
    slot_var = {}
    curr_name = None
    for s in str_repr.split(','):
        if '(' in s and ')' in s:
            slot_var[s.split('(')[0].strip()] = []
            var_name = s.split('(')[1].split(')')[0].strip()
            if var_name:
                slot_var[s.split('(')[0].strip()].append(var_name)
        elif '(' in s:
            slot_name, var_name = s.split('(')
            slot_var[slot_name.strip()] = [var_name.strip()]
            curr_name = slot_name.strip()
        elif ')' in s:
            var_name = s.split(')')[0]
            slot_var[curr_name].append(var_name.strip())
            curr_name = None
        elif curr_name:
            slot_var[curr_name].append(s.strip())
        else:
            slot_var[s.strip()] = []

    return slot_var


def rml_to_template(tree: RmlElement, namespace: RosemaryNamespace) -> RosemaryTemplate:
    variables = []
    if 'var' in tree.attributes:
        variables = list(map(str.strip, tree.attributes['var'].split(',')))

    slot_vars = []
    if 'slot' in tree.attributes:
        slot_vars = _get_slot_var(tree.attributes['slot'])

    return RosemaryTemplate(tree, variables, slot_vars, namespace)
