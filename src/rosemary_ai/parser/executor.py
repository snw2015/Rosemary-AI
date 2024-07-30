from abc import ABCMeta, abstractmethod
from typing import TypeAlias, Dict, Any, List, Tuple

from ..exceptions import RmlFormatException
from ..multi_modal.image import Image
from .data_expression import DataExpression
from .environment import VariableContext

IsPlainText: TypeAlias = bool
OutputValue: TypeAlias = str | Image
Value: TypeAlias = OutputValue | DataExpression
IsSucceed: TypeAlias = bool


class Executor(metaclass=ABCMeta):
    @abstractmethod
    def execute(self, value: Value, variables: VariableContext) -> IsSucceed:
        pass

    @abstractmethod
    def get_snapshot(self) -> Any:
        pass

    @abstractmethod
    def back_to_snapshot(self, state: Any):
        pass

    @abstractmethod
    def begin_scope(self, scope_name: str, key=None):
        pass

    @abstractmethod
    def end_scope(self, scope_name: str, succeed=True):
        pass


class FormatExecutor(Executor):
    def __init__(self):
        self.scope_stack: List[OutputValue | List | Dict | None] = [None]
        self.key_stack: List[str] = []

    def execute(self, value: Value, variables: VariableContext) -> IsSucceed:
        if isinstance(value, DataExpression):
            value = str(value.evaluate(variables))

        if self.scope_stack and isinstance(self.scope_stack[-1], list):
            if self.scope_stack[-1][-1] and isinstance(self.scope_stack[-1][-1], str) and isinstance(value, str):
                self.scope_stack[-1][-1] += value
            else:
                self.scope_stack[-1].append(value)

        elif not isinstance(value, Image):
            if self.scope_stack[-1] is None:
                self.scope_stack[-1] = value
            elif isinstance(self.scope_stack[-1], str) and isinstance(value, str):
                self.scope_stack[-1] += value
            else:
                self.scope_stack[-1] = value
        else:
            if self.scope_stack[-1] is None:
                self.scope_stack[-1] = [value]
            elif isinstance(self.scope_stack[-1], str):
                self.scope_stack[-1] = [self.scope_stack[-1], value]
            else:
                self.scope_stack[-1] = [value]

        return True

    def get_snapshot(self) -> Any:
        pass

    def back_to_snapshot(self, state: Any):
        pass

    def begin_scope(self, scope_type: str, key=None):
        if scope_type == 'list':
            assert self.scope_stack[-1] is None
            self.scope_stack[-1] = []
        elif scope_type == 'dict':
            assert self.scope_stack[-1] is None
            self.scope_stack[-1] = {}
        elif scope_type == 'list_item':
            if self.scope_stack[-1] is None or not isinstance(self.scope_stack[-1], list):
                raise RmlFormatException('"list-item" must be put directly under a "list".')
            self.scope_stack.append(None)
        elif scope_type == 'dict_item':
            if self.scope_stack[-1] is None or not isinstance(self.scope_stack[-1], dict):
                raise RmlFormatException('"dict-item" must be put directly under a "dict".')
            if key is None:
                raise RmlFormatException('"dict-item" must have a key which is not None.')

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


OUTPUT_INDICATOR = '__'


class ParseExecutor(Executor):
    def __init__(self, raw_str: str, target: str, target_obj, is_parse_strict: bool):
        self.raw_str: str = raw_str
        self.last_target_repr_with_var: Tuple[DataExpression, VariableContext] | None = None
        self.assign_with_var_list: List[Tuple[DataExpression, VariableContext]] = []
        self.target = target
        self.target_obj = target_obj
        self.is_parse_strict = is_parse_strict

    def execute(self, value: Value, variables: VariableContext) -> IsSucceed:
        if isinstance(value, DataExpression):
            value: DataExpression
            if self.last_target_repr_with_var is None:
                self.last_target_repr_with_var = (value, variables.copy())
        elif isinstance(value, str):
            if not self.is_parse_strict:
                value: str = value.strip()
            pos = self.raw_str.find(value)
            if pos == -1:
                return False
            leading = self.raw_str[:pos]
            if self.last_target_repr_with_var is not None:
                self._store_assignment(leading)
                self.last_target_repr_with_var = None
            self.raw_str = self.raw_str[pos + len(value):]
        else:
            assert False

        return True

    def get_snapshot(self) -> Tuple[str, Tuple[DataExpression, VariableContext] | None,
                                    List[Tuple[DataExpression, VariableContext]]]:

        return self.raw_str, self.last_target_repr_with_var, self.assign_with_var_list.copy()

    def back_to_snapshot(self, state: Tuple[str, Tuple[DataExpression, VariableContext] | None,
                         List[Tuple[DataExpression, VariableContext]]]):

        self.raw_str, self.last_target_repr_with_var, self.assign_with_var_list = state

    def begin_scope(self, scope_name: str, key=None):
        pass

    def end_scope(self, scope_name: str, succeed=True):
        pass

    def activate_assignments(self, assign_remain: bool):
        if self.last_target_repr_with_var is not None and assign_remain:
            self._store_assignment(self.raw_str)
        result = self.target_obj
        for assign, env in self.assign_with_var_list:
            if self.target:
                env[self.target] = result

            assign.execute(env, False)

            if self.target:
                result = env.get(self.target)

        return result

    def _store_assignment(self, value: str):
        target_repr, var = self.last_target_repr_with_var
        var[OUTPUT_INDICATOR] = value
        self.assign_with_var_list += [
            (target_repr, var)
        ]
