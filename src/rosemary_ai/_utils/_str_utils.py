from typing import Tuple


def full_name_to_indicator(full_name: str) -> Tuple[str, ...]:
    return tuple(full_name.split('.'))


def calc_leading_size(line: str) -> int:
    leading_size = 0
    while line.startswith(' ') or line.startswith('\t'):
        leading_size += 1 if line[0] == ' ' else 4
        line = line[1:]

    return leading_size


def clean_leading_ws_line(line: str, to_clean: int) -> str:
    leading_size = calc_leading_size(line)
    line = line.lstrip()
    if leading_size > to_clean:
        remain_size = leading_size - to_clean
        line = '\t' * (remain_size // 4) + ' ' * (remain_size % 4) + line

    return line


def clean_leading_ws_lines(text: str, to_clean: int) -> str:
    if to_clean == 0:
        return text
    lines = text.splitlines()
    cleaned_lines = []
    if lines and lines[0].strip():
        cleaned_lines += [lines[0]]
        lines = lines[1:]
    for line in lines:
        cleaned_lines += [clean_leading_ws_line(line, to_clean)]

    return '\n'.join(cleaned_lines)


def calc_leading_ws_and_remove_leading(text: str) -> Tuple[int, str]:
    if not text:
        return 0, ''
    lines = list(reversed(text.splitlines()))
    line = ''
    while lines and not lines[-1].strip():
        line = lines.pop()

    if lines:
        line = lines.pop()

    leading_size = calc_leading_size(line)

    lines += [line]

    return (leading_size,
            '\n'.join(clean_leading_ws_line(line, leading_size) for line in reversed(lines)))


def remove_trailing_blank_lines(text: str) -> str:
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return '\n'.join(lines)
