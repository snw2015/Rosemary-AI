from typing import TypeAlias, Dict, Any

VariableContext: TypeAlias = Dict[str, Any]


class DataExpression:
    def __init__(self, value: str):
        self._value = value

    def value(self):
        return self._value

    def evaluate(self, context: VariableContext, need_copy=True):
        # The eval function is destructive.
        # For time complexity issues, the caller should decide when to not copy the context.
        if need_copy:
            return eval(self._value, context.copy())
        else:
            return eval(self._value, context)

    def execute(self, context: VariableContext, need_copy=True):
        if need_copy:
            exec(self._value, context.copy())
        else:
            exec(self._value, context)

    def __add__(self, other) -> 'DataExpression':
        if isinstance(other, str):
            return DataExpression(self._value + other)
        if not isinstance(other, DataExpression):
            return DataExpression(self._value + f'{other}')
        return DataExpression(self._value + other._value)
