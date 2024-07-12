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
