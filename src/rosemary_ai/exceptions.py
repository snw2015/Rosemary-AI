class RmlTagNotClosedException(Exception):
    def __init__(self, tag_name: str, wrong_tag_name: str):
        super().__init__(f'Tag "{tag_name}" is not closed properly. Found "{wrong_tag_name}" instead.')


class RmlSyntaxException(Exception):
    def __init__(self, message, src_path):
        super().__init__(f'{message} in "{src_path}".')


class RmlFormatException(Exception):
    def __init__(self, message):
        super().__init__(message)


class ExecutionException(Exception):
    def __init__(self, message):
        super().__init__(message)


class ParsingFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)


class RequestFailedException(Exception):
    def __init__(self, response):
        super().__init__(response.text)
