def escape_data_indicator(raw: str) -> str:
    return raw.replace('{{', '{').replace('}}', '}')


def escape_plain_text(raw: str) -> str:
    return (raw.
            replace('<<', '<').
            replace('>>', '>').
            replace('{{', '{').
            replace('}}', '}')
            )


def escape_attribute_value(raw: str) -> str:
    return raw.replace('\\"', '"').replace("\\\\", "\\")