from __future__ import annotations

import logging

DEBUGFORMATTER = '[%(asctime)s] [%(levelname)s] (%(filename)s:%(lineno)s) \t- %(message)s'


class CustomConsoleFormatter(logging.Formatter):
    reset = '\x1b[0m'
    bold = '\x1b[1m'
    italic = '\x1b[3m'
    reverse = '\x1b[7m'

    white = f'{reset}\x1b[38;2;255;255;255m'
    bold_white = f'{reset}{bold}\x1b[38;2;255;255;255m'
    yellow = f'{reset}\x1b[38;2;255;255;95m'
    italic_yellow = f'{reset}{italic}\x1b[38;2;255;255;95m'
    bold_yellow = f'{reset}{bold}\x1b[38;2;255;255;95m'
    mintgreen = f'{reset}\x1b[38;2;153;255;153m'
    bold_mintgreen = f'{reset}{bold}\x1b[38;2;153;255;153m'
    red = f'{reset}\x1b[38;2;255;127;80m'
    bold_red = f'{reset}{bold}\x1b[38;2;255;127;80m'
    rev_bold_red = f'{reset}{reverse}{bold}\x1b[38;2;255;127;80m'

    CONSOLEFORMATTER = '{}%(module)12s:  {}[%(levelname)-.3s]  {}%(message)s{}'

    FORMATS = {
        logging.DEBUG: CONSOLEFORMATTER.format(white, white, white, reset),
        logging.INFO: CONSOLEFORMATTER.format(white, bold_mintgreen, white, reset),
        logging.WARNING: CONSOLEFORMATTER.format(white, bold_yellow, italic_yellow, reset),
        logging.ERROR: CONSOLEFORMATTER.format(white, bold_red, red, reset),
        logging.CRITICAL: CONSOLEFORMATTER.format(white, rev_bold_red, rev_bold_red, reset),
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
