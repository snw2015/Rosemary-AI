from abc import ABCMeta, abstractmethod
from typing import TypeAlias, Dict, Any, List, Tuple

from ..multi_modal.image import Image
from .data_expression import DataExpression
from .leaf_elements import VariableContext

IsPlainText: TypeAlias = bool
Value: TypeAlias = str | DataExpression | Image
IsSucceed: TypeAlias = bool


class Executor(metaclass=ABCMeta):
    @abstractmethod
    def execute(self, value: Value, variables: VariableContext) -> IsSucceed:
        pass

    @abstractmethod
    def get_snapshot(self) -> Any:
        pass

    @abstractmethod
    def set_snapshot(self, state: Any):
        pass

    @abstractmethod
    def begin_scope(self, scope_name: str, key=None):
        pass

    @abstractmethod
    def end_scope(self, scope_name: str, succeed=True):
        pass


class FormatExecutor(Executor):
    def __init__(self):
        self.scope_stack: List[None | str | List | Dict] = [None]
        self.key_stack: List[str] = []

    def execute(self, value: Value, variables: VariableContext) -> IsSucceed:
        if isinstance(value, DataExpression):
            value: DataExpression
            value = value.evaluate(variables)

        if self.scope_stack[-1] is None:
            self.scope_stack[-1] = []
        self.scope_stack[-1] += [value]

        return True

    def get_snapshot(self) -> Any:
        pass

    def set_snapshot(self, state: Any):
        pass

    def begin_scope(self, scope_type: str, key=None):
        if scope_type == 'list':
            assert self.scope_stack[-1] is None
            self.scope_stack[-1] = []
        elif scope_type == 'dict':
            assert self.scope_stack[-1] is None
            self.scope_stack[-1] = {}
        elif scope_type == 'list_item':
            assert self.scope_stack[-1] is not None
            self.scope_stack.append(None)
        elif scope_type == 'dict_item':
            assert self.scope_stack[-1] is not None
            assert key is not None
            self.key_stack.append(key)
            self.scope_stack.append(None)
        else:
            assert False

    def end_scope(self, scope_type: str, succeed=True):
        if scope_type == 'list' or scope_type == 'dict':
            return
        obj = self.scope_stack.pop()
        assert self.scope_stack
        if not succeed:  # simply not put the obj into the parent container
            return
        if scope_type == 'list_item':
            assert isinstance(self.scope_stack[-1], list)
            self.scope_stack[-1] += [obj]
        elif scope_type == 'dict_item':
            assert isinstance(self.scope_stack[-1], dict)
            self.scope_stack[-1][self.key_stack.pop()] = obj
        else:
            assert False

    def get_result(self):
        return self.scope_stack[0]


class ParseExecutor(Executor):
    def __init__(self, raw_str: str):
        self.raw_str: str = raw_str
        self.last_target_repr_with_var: Tuple[DataExpression, VariableContext] | None = None
        self.assign_with_var_list: List[Tuple[DataExpression, VariableContext]] = []

    def execute(self, value: Value, variables: VariableContext) -> IsSucceed:
        if isinstance(value, DataExpression):
            value: DataExpression
            if self.last_target_repr_with_var is None:
                self.last_target_repr_with_var = (value, variables.copy())
        elif isinstance(value, str):
            value: str
            pos = self.raw_str.find(value)
            if pos == -1:
                return False
            leading = self.raw_str[:pos]
            if self.last_target_repr_with_var is not None:
                target_repr, var = self.last_target_repr_with_var
                self.assign_with_var_list += [
                    (target_repr + f'={repr(leading)}', var)
                ]
                self.last_target_repr_with_var = None
            self.raw_str = self.raw_str[pos + len(value):]
        else:
            assert False

        return True

    def get_snapshot(self) -> Tuple[str, Tuple[DataExpression, VariableContext] | None,
            List[Tuple[DataExpression, VariableContext]]]:

        return self.raw_str, self.last_target_repr_with_var, self.assign_with_var_list.copy()

    def set_snapshot(self, state: Tuple[str, Tuple[DataExpression, VariableContext] | None,
            List[Tuple[DataExpression, VariableContext]]]):

        self.raw_str, self.last_target_repr_with_var, self.assign_with_var_list = state

    def begin_scope(self, scope_name: str, key=None):
        pass

    def end_scope(self, scope_name: str, succeed=True):
        pass

    def activate_assignments(self, assign_remain: bool):
        if self.last_target_repr_with_var and assign_remain:
            target_repr, var = self.last_target_repr_with_var
            self.assign_with_var_list += [(target_repr + f'{repr(self.raw_str)}', var)]
        for assign, env in self.assign_with_var_list:
            assign.execute(env, False)
