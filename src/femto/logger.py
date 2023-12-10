from __future__ import annotations

import logging

DEBUGFORMATTER = '[%(asctime)s] [%(levelname)s] (%(filename)s:%(lineno)s) \t- %(message)s'


class CustomConsoleFormatter(logging.Formatter):
    white = '\033[0;97m'
    cyan = '\033[0;96m'
    yellow = '\033[0;93m'
    bold_yellow = '\033[1;93m'
    blue = '\033[0;94m'
    red = '\033[0;91m'
    bold_red = '\033[1;91m'
    reset = '\033[0m'

    CONSOLEFORMATTER = '{}%(filename)16s:  {}[%(levelname)-.3s]  {}%(message)s{}'

    FORMATS = {
        logging.DEBUG: CONSOLEFORMATTER.format(white, white, white, reset),
        logging.INFO: CONSOLEFORMATTER.format(white, white, white, reset),
        logging.WARNING: CONSOLEFORMATTER.format(yellow, bold_yellow, yellow, reset),
        logging.ERROR: CONSOLEFORMATTER.format(white, bold_red, red, reset),
        logging.CRITICAL: CONSOLEFORMATTER.format(white, bold_red, bold_red, reset),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
