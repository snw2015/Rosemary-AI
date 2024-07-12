from typing import TypeAlias, Dict, Any

from ..exceptions import ExecutionException

VariableContext: TypeAlias = Dict[str, Any]


class DataExpression:
    def __init__(self, value: str):
        self._value = value

    def value(self):
        return self._value

    def evaluate(self, context: VariableContext, need_copy=True):
        # The eval function is destructive to the context dict.
        # For time complexity issues, the caller should decide when to not copy the context.
        try:
            if need_copy:
                return eval(self._value, context.copy())
            else:
                return eval(self._value, context)
        except Exception as e:
            raise ExecutionException(f'Failed to evaluate Python code "{self._value}": {e}.')

    def execute(self, context: VariableContext, need_copy=True):
        try:
            if need_copy:
                exec(self._value, context.copy())
            else:
                exec(self._value, context)
        except Exception as e:
            raise ExecutionException(f'Failed to execute Python code "{self._value}": {e}.')

    def __str__(self):
        return f'DataExpression<{self._value}>'

    def __repr__(self):
        return self.__str__()
