from abc import ABC, abstractmethod
from typing import TypeAlias, Generator, TypeVar, Union, Generic, Dict, Any, AsyncIterable

from ..multi_modal.image import Image
from .api_key_manager import get_api_key

CONTENT_TYPES = (str, Image, bytes)
ContentType: TypeAlias = Union[*CONTENT_TYPES]
T = TypeVar('T', *CONTENT_TYPES)  # noqa


class AbstractContentGenerator(ABC, Generic[T]):
    def __init__(self, provider: str):
        self.provider = provider

    def get_api_key(self, api_key_overridden: str) -> str:
        if api_key_overridden is None:
            api_key_overridden = get_api_key(self.provider)

        return api_key_overridden

    @abstractmethod
    def generate(self, data, options: Dict[str, Any], dry_run: bool, api_key: str = None) -> T:
        pass

    @abstractmethod
    async def generate_async(self, data, options: Dict[str, Any], dry_run: bool, api_key: str = None) -> T:
        pass

    ####
    # Stream should be generated incrementally. E.g.: '', 'this', 'this is', 'this is a', 'this is a test'
    ###
    @abstractmethod
    def generate_stream(self, data, options: Dict[str, Any],
                        dry_run: bool, api_key: str = None) -> Generator[T, None, None]:
        pass

    @abstractmethod
    def generate_stream_async(self, data, options: Dict[str, Any],
                              dry_run: bool, api_key: str = None) -> AsyncIterable[T]:
        pass
