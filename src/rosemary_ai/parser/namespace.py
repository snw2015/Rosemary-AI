from typing import Dict, Tuple, TypeVar, Generic, List


T = TypeVar('T')


class Namespace(Generic[T]):
    def __init__(self, parent: 'Namespace' = None):
        self._local: Dict[str, T | 'Namespace'] = {}
        self._parent = parent

    def __getitem__(self, key: str):
        if key in self._local:
            return self._local[key]
        if self._parent is not None:
            return self._parent[key]
        raise KeyError(f'Key {key} not found')

    def append(self, key: str, value: T | 'Namespace'):
        self._local[key] = value

    def get_by_indicator(self, indicator: Tuple[str, ...]) -> T | 'Namespace':
        assert indicator
        if len(indicator) == 1:
            return self[indicator[0]]
        else:
            return self[indicator[0]].get_by_indicator(indicator[1:])

    def items(self) -> List[Tuple[str, T | 'Namespace']]:
        return list(self._local.items())

    def __str__(self):
        return f'Namespace{self._local}'

    def __repr__(self):
        return self.__str__()
