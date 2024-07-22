from typing import Dict

_API_KEYS: Dict[str, str] = {}


def get_api_key(provider: str) -> str:
    return _API_KEYS.get(provider, None)


def set_api_key(provider: str, api_key: str):
    _API_KEYS[provider] = api_key


def api_keys() -> Dict[str, str]:
    return _API_KEYS.copy()
