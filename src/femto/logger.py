from __future__ import annotations

import logging

DEBUGFORMATTER = '[%(asctime)s] [%(levelname)s] (%(filename)s:%(lineno)s) \t- %(message)s'


class CustomConsoleFormatter(logging.Formatter):

    bold = '\x1b[1m'
    reset = '\x1b[0m'

    white = f'{reset}\x1b[38;2;255;255;255m'
    bold_white = f'{bold}\x1b[38;2;255;255;255m'
    yellow = f'{reset}\x1b[38;2;255;255;95m'
    bold_yellow = f'{bold}\x1b[38;2;255;255;95m'
    mintgreen = f'\x1b[38;2;153;255;153m'
    bold_mintgreen = f'{bold}\x1b[38;2;153;255;153m'
    red = f'\x1b[38;2;255;127;80m'
    bold_red = f'{bold}\x1b[38;2;255;127;80m'
    rev_bold_red = f'\x1b[7m{bold}38;2;255;127;80m'

    CONSOLEFORMATTER = '{}%(module)12s:  {}[%(levelname)-.3s]  {}%(message)s'

    FORMATS = {
        logging.DEBUG: CONSOLEFORMATTER.format(white, white, white),
        logging.INFO: CONSOLEFORMATTER.format(white, bold_mintgreen, white),
        logging.WARNING: CONSOLEFORMATTER.format(white, bold_yellow, yellow),
        logging.ERROR: CONSOLEFORMATTER.format(white, bold_red, red),
        logging.CRITICAL: CONSOLEFORMATTER.format(white, rev_bold_red, rev_bold_red),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
