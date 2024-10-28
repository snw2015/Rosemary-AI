from enum import Enum
from typing import List, Tuple

from lark import Transformer

from .._utils.str_escape import escape_data_indicator, escape_attribute_value, escape_plain_text  # noqa
from .._utils.str_utils import (calc_leading_ws_and_remove_leading, clean_leading_ws_lines, # noqa
                                remove_trailing_blank_lines)
from ..exceptions import RmlTagNotClosedException


class RmlElement:
    def __init__(self, is_text: bool, indicator: Tuple[str, ...], text_tokens=None, children=None, attributes=None):
        if text_tokens is None:
            text_tokens = []
        if attributes is None:
            attributes = {}
        if children is None:
            children = []
        self.is_text = is_text
        self.indicator = indicator
        self.text_tokens = text_tokens
        self.children = children
        self.attributes = attributes

    def __str__(self):
        return f'<{self.indicator}@{self.attributes}>{self.text_tokens if self.is_text else self.children}'

    def __repr__(self):
        return self.__str__()


class TextToken:
    class TYPE(Enum):
        PLAIN_TEXT = 1
        INDICATOR = 2

    def __init__(self, text_type: TYPE, text: str):
        self.type = text_type
        self.text = text

    def __str__(self):
        if self.type == TextToken.TYPE.PLAIN_TEXT:
            return f'"{self.text}"'
        else:
            return f'${self.text}'

    def __repr__(self):
        return self.__str__()


def cleandoc(items: List[TextToken]):
    if not items:
        return []
    cleaned = []

    items = list(reversed(items))

    while items and items[-1].type == TextToken.TYPE.INDICATOR:
        cleaned += [items.pop()]
    if not items:
        return cleaned
    leading_ws, non_empty_part = calc_leading_ws_and_remove_leading(items.pop().text)
    if non_empty_part:
        cleaned += [TextToken(TextToken.TYPE.PLAIN_TEXT, non_empty_part)]

    for item in reversed(items):
        if item.type == TextToken.TYPE.INDICATOR:
            cleaned += [item]
        else:
            cleaned_line = clean_leading_ws_lines(item.text, leading_ws)
            if cleaned_line:
                cleaned += [TextToken(TextToken.TYPE.PLAIN_TEXT, cleaned_line)]

    if cleaned and cleaned[-1].type == TextToken.TYPE.PLAIN_TEXT:
        cleaned[-1].text = remove_trailing_blank_lines(cleaned[-1].text)

    return cleaned


class TreeToRmlTreeTransformer(Transformer):
    def rosemary(self, items):  # noqa
        element = RmlElement(False, ('$rosemary',))
        element.children = [item for item in items if item]
        return element

    def element_without_body(self, items):  # noqa
        element = RmlElement(False, items[0])
        element.attributes = items[1]
        return element

    def element_with_body(self, items):  # noqa
        if items[0] != items[3]:
            raise RmlTagNotClosedException('.'.join(items[0]), '.'.join(items[3]))

        element = RmlElement(False, items[0])
        element.attributes = items[1]
        element.children = [child for child in items[2].children if child is not None]

        return element

    def element_indicator(self, items: List[str]) -> Tuple[str, ...]:  # noqa
        return tuple(items)

    def attributes(self, items):  # noqa
        return dict(items)

    def attribute_with_value(self, items):  # noqa
        if len(items) == 1:
            items += ['']
        return items

    def attribute_without_value(self, items):  # noqa
        return items[0], 'True'

    def xml_text(self, items):  # noqa
        if not items:
            return None
        if len(items) == 1 and items[0].type == TextToken.TYPE.PLAIN_TEXT and not items[0].text.strip():
            return None
        tokens = cleandoc(items)
        element = RmlElement(True, ('$text',), text_tokens=tokens)
        return element

    def ignore_text(self, items):  # noqa
        text = items[0] if items else ''
        return TextToken(TextToken.TYPE.PLAIN_TEXT, text)

    def plain_text(self, items):  # noqa
        return TextToken(TextToken.TYPE.PLAIN_TEXT, escape_plain_text(items[0]))

    def placeholder(self, items):  # noqa
        text = items[0] if items else ''
        return TextToken(TextToken.TYPE.INDICATOR, text)

    def COMMENT(self, token):  # noqa
        return None

    def DATA_INDICATOR(self, token):  # noqa
        return escape_data_indicator(token.value)

    def INDICATOR_NAME(self, token):  # noqa
        return token.value

    def ATTRIBUTE_VALUE(self, token):  # noqa
        return escape_attribute_value(token.value)
