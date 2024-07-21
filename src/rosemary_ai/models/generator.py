from abc import ABC, abstractmethod
from typing import TypeAlias, Generator, TypeVar, Union, Generic, Dict, Any, Callable, AsyncIterable

from ..multi_modal.image import Image

CONTENT_TYPES = (str, Image)
ContentType: TypeAlias = Union[*CONTENT_TYPES]
T = TypeVar('T', *CONTENT_TYPES)  # noqa


class AbstractContentGenerator(ABC, Generic[T]):
    @abstractmethod
    def generate(self, data, options: Dict[str, Any], dry_run: bool) -> T:
        pass

    @abstractmethod
    async def generate_async(self, data, options: Dict[str, Any], dry_run: bool) -> T:
        pass

    ####
    # Stream should be generated incrementally. E.g.: '', 'this', 'this is', 'this is a', 'this is a test'
    ###
    @abstractmethod
    def generate_stream(self, data, options: Dict[str, Any], dry_run: bool) -> Generator[T, None, None]:
        pass

    @abstractmethod
    def generate_stream_async(self, data, options: Dict[str, Any],
                                dry_run: bool) -> AsyncIterable[T]:
        pass
