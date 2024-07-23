from typing import Dict, Any


def options_with_default(options: Dict[str, Any], default_options: Dict[str, Any]) -> Dict[str, Any]:
    if default_options is None:
        return options
    if options is None:
        return default_options

    return default_options | options
