import logging


class RosemaryLogger:
    def __init__(self):
        self.logger = logging.getLogger('rosemary')
        self.verbose = False

    def debug(self, msg: str):
        if self.verbose:
            self.logger.debug(msg)

    def info(self, msg: str):
        if self.verbose:
            self.logger.info(msg)

    def warning(self, msg: str):
        if self.verbose:
            self.logger.warning(msg)

    def error(self, msg: str):
        if self.verbose:
            self.logger.error(msg)

    def set_verbose(self, verbose: int):
        self.verbose = verbose

    def set_logger(self, logger: logging.Logger):
        self.logger = logger

    def set_logging_level(self, level: int):
        self.logger.setLevel(level)


LOGGER = RosemaryLogger()
LOGGER.set_logging_level(logging.INFO)


def set_verbose(verbose: bool):
    LOGGER.set_verbose(verbose)


def set_logger(logger: logging.Logger):
    LOGGER.set_logger(logger)


def set_logging_level(level: int):
    LOGGER.set_logging_level(level)

