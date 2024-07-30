from typing import Generator, Dict, Any, List, Tuple, Callable, TypeVar, Generic

import httpx
import requests
from httpx import AsyncClient

from ._utils import update_options, check_response_status
from .generator import AbstractContentGenerator
from ..exceptions import RmlFormatException

from .._logger import LOGGER


AUTH_METHODS = ('Bearer',)
T = TypeVar('T')


def _generate_auth(auth_method: str, api_key: str):
    if auth_method == 'Bearer':
        return f'Bearer {api_key}'
    else:
        raise RmlFormatException(f'Unsupported authentication method: {auth_method}. '
                                 f'Supported methods: {AUTH_METHODS}.')


class RequestGenerator(AbstractContentGenerator[T], Generic[T]):
    def __init__(self, url: str, method: str = 'POST', auth_method: str = 'Bearer',
                 accept_type: str = 'application/json', provider: str = None,
                 post_handle: Callable[[bytes], T] = None):
        super().__init__(provider)
        self.url = url
        self.method = method
        self.auth_method = auth_method
        self.accept_type = accept_type
        if post_handle:
            self.post_handle = post_handle

    def post_handle(self, response_content: bytes) -> T:
        return response_content

    def _set_up(self, data: Dict[str, Any],
                options: Dict[str, Any], dry_run: bool, api_key: str) -> Tuple:
        headers = {
            'accept': self.accept_type,
            'authorization': _generate_auth(self.auth_method, self.get_api_key(api_key))
        }

        update_options(options, data['data'])
        json_data = options

        files_obj = data['files']
        if files_obj:
            files = [(name, open(path, 'rb')) for name, path in files_obj.items()]
        else:
            files = None

        LOGGER.info(f'Sending data to {self.url}.')
        LOGGER.info(f'Files: {files}.')
        LOGGER.info(f'JSON data: {json_data}.')

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')

        timeout = options.pop('timeout', None)

        return headers, files, json_data, timeout

    def generate(self, data: Dict[str, str | List[str]],
                 options: Dict[str, Any], dry_run: bool, api_key: str = None) -> T:
        headers, files, json_data, timeout = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return None

        if files:
            response = requests.request(
                method=self.method,
                url=self.url,
                headers=headers,
                files=files,
                data=json_data,
                timeout=timeout
            )
        else:
            response = requests.request(
                method=self.method,
                url=self.url,
                headers=headers,
                json=json_data,
                timeout=timeout
            )

        LOGGER.info(f'Received response from {self.url}: "{response}".')

        check_response_status(response)

        return self.post_handle(response.content)

    async def generate_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                             options: Dict[str, Any], dry_run: bool, api_key: str = None) -> T:
        headers, files, json_data, timeout = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return None

        async with AsyncClient() as client:

            if files:
                response: httpx.Response = await client.request(
                    method=self.method,
                    url=self.url,
                    headers=headers,
                    files=files,
                    data=json_data,
                    timeout=timeout
                )
            else:
                response: httpx.Response = await client.request(
                    method=self.method,
                    url=self.url,
                    headers=headers,
                    json=json_data,
                    timeout=timeout
                )

        LOGGER.info(f'Received response from {self.url}: "{response}".')

        check_response_status(response)

        return self.post_handle(response.content)

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]],
                        options: Dict[str, Any],
                        dry_run: bool, api_key: str = None) -> Generator[T, None, None]:
        raise NotImplementedError('Stream generation is not supported for general HTTP request.')

    async def generate_stream_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                                    options: Dict[str, Any],
                                    dry_run: bool, api_key: str = None) -> Generator[T, None, None]:
        raise NotImplementedError('Stream generation is not supported for general HTTP request.')
