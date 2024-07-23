from typing import Dict, Tuple, TypeVar, Generic, List, Iterable

from .._utils.str_utils import full_name_to_indicator

T = TypeVar('T')


class Namespace(Generic[T]):
    def __init__(self, parent: 'Namespace' = None):
        self._local: Dict[str, T | 'Namespace'] = {}
        self._parent = parent

    def __getitem__(self, full_name_or_names: str | Iterable[str]):
        if isinstance(full_name_or_names, str):
            return self[full_name_to_indicator(full_name_or_names)]

        return self._get_by_indicator((*full_name_or_names,))

    def _get_by_name(self, name: str):
        if name in self._local:
            return self._local[name]
        if self._parent is not None:
            return self._parent[name]
        raise KeyError(f'Key {name} not found in namespace.')

    def append(self, key: str, value: T | 'Namespace'):
        self._local[key] = value

    def _get_by_indicator(self, indicator: Tuple[str, ...]) -> T | 'Namespace':
        assert indicator
        if len(indicator) == 1:
            return self._get_by_name(indicator[0])
        else:
            return self._get_by_name(indicator[0])._get_by_indicator(indicator[1:])

    def items(self) -> List[Tuple[str, T | 'Namespace']]:
        return list(self._local.items())

    def __str__(self):
        return f'Namespace{self._local}'

    def __repr__(self):
        return self.__str__()
