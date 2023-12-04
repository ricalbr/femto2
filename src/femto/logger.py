from __future__ import annotations

import logging

CONSOLEFORMATTER = '%(filename)16s:  [%(levelname)-.3s]  %(message)s'
DEBUGFORMATTER = '[%(asctime)s] [%(levelname)s] (%(filename)s:%(lineno)s) \t- %(message)s'


class CustomConsoleFormatter(logging.Formatter):

    grey = '\x1b[38;20m'
    yellow = '\x1b[33;20m'
    bold_yellow = '\033[1;33m'
    red = '\x1b[31;20m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'
    format = CONSOLEFORMATTER

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: bold_yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
