from typing import Generator, Dict, Any, List, Tuple

import httpx
import requests
from httpx import AsyncClient

from ._utils import update_options, check_response_status
from .generator import AbstractContentGenerator
from ..exceptions import RmlFormatException

from .._logger import LOGGER

_HOST = 'https://api.stability.ai'


class StabilityImageGenerator(AbstractContentGenerator[bytes]):
    _URL_OF_MODEL_NAME = {
        'stable-diffusion-ultra': '/v2beta/stable-image/generate/ultra',
        'stable-diffusion-core': '/v2beta/stable-image/generate/core',
        'stable-diffusion-3-large': '/v2beta/stable-image/generate/sd3',
        'stable-diffusion-3-large-turbo': '/v2beta/stable-image/generate/sd3',
        'stable-diffusion-3-medium': '/v2beta/stable-image/generate/sd3',
    }

    _SD3_MODEL_OF_MODEL_NAME = {
        'stable-diffusion-3-large': 'sd3-large',
        'stable-diffusion-3-large-turbo': 'sd3-large-turbo',
        'stable-diffusion-3-medium': 'sd3-medium',
    }

    def __init__(self, model_name: str):
        super().__init__('Stability')
        self.model_name = model_name

    def _set_up(self, data: Dict[str, str | List[str | Dict[str, str | List]]],
                options: Dict[str, Any], dry_run: bool, api_key: str) -> Tuple:
        prompt = data.pop('prompt')
        if isinstance(prompt, list):
            raise RmlFormatException('Prompt must only contain string.')

        data: Dict[str, List[str]]
        # update_options(options, data, STABLE_GEN_V2_OPTION_TYPES)
        update_options(options, data)

        LOGGER.info(f'Sending prompt to {self.model_name}: "{prompt}".')
        LOGGER.debug(f'Options: {options}.')

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')

        api_key = self.get_api_key(api_key)

        model = self._SD3_MODEL_OF_MODEL_NAME.get(self.model_name, None)

        options['model'] = model

        timeout = options.pop('timeout', None)

        url = _HOST + self._URL_OF_MODEL_NAME[self.model_name]

        return prompt, options, api_key, url, timeout

    def generate(self, data: Dict[str, str | List[str]],
                 options: Dict[str, Any], dry_run: bool, api_key: str = None) -> bytes:
        prompt, options, api_key, url, timeout = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return b''

        response = requests.post(
            url=url,
            headers={
                'authorization': f'Bearer {api_key}',
                'accept': 'image/*'
            },
            files={'none': ''},
            data={
                'prompt': prompt,
                **options
            },
            timeout=timeout
        )

        LOGGER.info(f'Received response from {self.model_name}: "{response}".')

        check_response_status(response)

        return response.content

    async def generate_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                             options: Dict[str, Any], dry_run: bool, api_key: str = None) -> bytes:
        prompt, options, api_key, url, timeout = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return b''

        async with AsyncClient() as client:
            response: httpx.Response = await client.post(
                url=url,
                headers={
                    'authorization': f'Bearer {api_key}',
                    'accept': 'image/*'
                },
                files={'none': ''},
                data={
                    'prompt': prompt,
                    **options
                },
                timeout=timeout
            )

        LOGGER.info(f'Received response from {self.model_name}: "{response}".')

        check_response_status(response)

        return response.content

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]],
                        options: Dict[str, Any],
                        dry_run: bool, api_key: str = None) -> Generator[bytes, None, None]:
        raise NotImplementedError('Stream generation is not supported for image generation.')

    async def generate_stream_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                                    options: Dict[str, Any],
                                    dry_run: bool, api_key: str = None) -> Generator[bytes, None, None]:
        raise NotImplementedError('Stream generation is not supported for image generation.')


class StabilityV1ImageGenerator(AbstractContentGenerator[bytes]):
    _ENGINE_ID_OF_MODEL_NAME = {
        'stable-diffusion-xl': 'stable-diffusion-xl-1024-v1-0',
        'stable-diffusion-1.6': 'stable-diffusion-v1-6',
        'stable-diffusion-beta': 'stable-diffusion-xl-beta-v2-2-2',
    }

    def __init__(self, model_name: str):
        super().__init__('Stability')
        self.model_name = model_name

    def _set_up(self, data: Dict[str, Any],
                options: Dict[str, Any], dry_run: bool, api_key: str) -> Tuple:
        prompts = data.pop('prompts')
        for prompt in prompts:
            if isinstance(prompt['text'], list):
                raise RmlFormatException('Prompt must only contain string.')
            if 'weight' in prompt:
                prompt['weight'] = float(prompt['weight'])

        data: Dict[str, List[str]]
        # update_options(options, data, STABLE_GEN_V1_OPTION_TYPES)
        update_options(options, data)

        LOGGER.info(f'Sending prompt to {self.model_name}: "{prompts}".')
        LOGGER.debug(f'Options: {options}.')

        if dry_run:
            LOGGER.info('Dry run mode enabled. Skipping API call.')

        api_key = self.get_api_key(api_key)

        assert self.model_name in self._ENGINE_ID_OF_MODEL_NAME
        engine_id = self._ENGINE_ID_OF_MODEL_NAME[self.model_name]

        url = _HOST + '/v1/generation/' + engine_id + '/text-to-image'

        timeout = options.pop('timeout', None)

        return prompts, options, api_key, url, timeout

    def generate(self, data: Dict[str, str | List[str]],
                 options: Dict[str, Any], dry_run: bool, api_key: str = None) -> bytes:
        prompts, options, api_key, url, timeout = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return b''

        response = requests.post(
            url=url,
            headers={
                'content-Type': 'application/json',
                'authorization': f'Bearer {api_key}',
                'accept': 'image/png'
            },
            json={
                'text_prompts': prompts,
                **options
            },
            timeout=timeout
        )

        LOGGER.info(f'Received response from {self.model_name}: "{response}".')

        check_response_status(response)

        return response.content

    async def generate_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                             options: Dict[str, Any], dry_run: bool, api_key: str = None) -> bytes:
        prompts, options, api_key, url, timeout = self._set_up(data, options, dry_run, api_key)

        if dry_run:
            return b''

        async with AsyncClient() as client:
            response: httpx.Response = await client.post(
                url=url,
                headers={
                    'content-type': 'application/json',
                    'authorization': f'Bearer {api_key}',
                    'accept': 'image/png'
                },
                json={
                    'text_prompts': prompts,
                    **options
                },
                timeout=timeout
            )

        LOGGER.info(f'Received response from {self.model_name}: "{response}".')

        check_response_status(response)

        return response.content

    def generate_stream(self, data: Dict[str, str | List[Dict[str, str | List]]],
                        options: Dict[str, Any],
                        dry_run: bool, api_key: str = None) -> Generator[bytes, None, None]:
        raise NotImplementedError('Stream generation is not supported for image generation.')

    async def generate_stream_async(self, data: Dict[str, str | List[Dict[str, str | List]]],
                                    options: Dict[str, Any],
                                    dry_run: bool, api_key: str = None) -> Generator[bytes, None, None]:
        raise NotImplementedError('Stream generation is not supported for image generation.')
