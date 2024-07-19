from typing import List, Tuple, TypeAlias, Dict

from ..parser.data_expression import VariableContext, DataExpression
from ..parser.leaf_elements import RosemaryPetal, RosemaryTemplate, RosemaryNamespace
from ..parser.transformer import RmlElement


class Slot:
    def __init__(self, elements_with_info: List[Tuple[RmlElement, 'Environment', VariableContext]],
                 parameter_names: List[str], is_inf: bool = False):
        self.element_with_info = elements_with_info
        self.parameter_names = parameter_names
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


def build_environment(petal, data: VariableContext) -> Environment:
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
    parameter_names = []

    # Deprecated
    if 'var' in tree.attributes:
        parameter_names = list(map(str.strip, tree.attributes['var'].split(',')))

    if 'param' in tree.attributes:
        parameter_names = list(map(str.strip, tree.attributes['param'].split(',')))

    return RosemaryPetal(formatter, parser, namespace, parameter_names, target,
                         tree.attributes['init'] if 'init' in tree.attributes else '{}')


def _get_slot_params(str_repr: str) -> Dict[str, List[str]]:
    slot_params = {}
    curr_name = None
    for s in str_repr.split(','):
        if '(' in s and ')' in s:
            slot_params[s.split('(')[0].strip()] = []
            param_name = s.split('(')[1].split(')')[0].strip()
            if param_name:
                slot_params[s.split('(')[0].strip()].append(param_name)
        elif '(' in s:
            slot_name, param_name = s.split('(')
            slot_params[slot_name.strip()] = [param_name.strip()]
            curr_name = slot_name.strip()
        elif ')' in s:
            param_name = s.split(')')[0]
            slot_params[curr_name].append(param_name.strip())
            curr_name = None
        elif curr_name:
            slot_params[curr_name].append(s.strip())
        else:
            slot_params[s.strip()] = []

    return slot_params


def rml_to_template(tree: RmlElement, namespace: RosemaryNamespace) -> RosemaryTemplate:
    parameter_names = []

    # Deprecated
    if 'var' in tree.attributes:
        parameter_names = list(map(str.strip, tree.attributes['var'].split(',')))

    if 'param' in tree.attributes:
        parameter_names = list(map(str.strip, tree.attributes['param'].split(',')))

    slot_params = {}
    if 'slot' in tree.attributes:
        slot_params = _get_slot_params(tree.attributes['slot'])

    return RosemaryTemplate(tree, parameter_names, slot_params, namespace)
