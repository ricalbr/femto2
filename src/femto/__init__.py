from __future__ import annotations

import logging

from femto.logger import CONSOLEFORMATTER
from femto.logger import DEBUGFORMATTER

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter(CONSOLEFORMATTER))
logger.addHandler(_ch)

_fh = logging.FileHandler('log.log', mode='w')
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter(DEBUGFORMATTER, '%Y-%m-%d %H:%M:%S'))
logger.addHandler(_fh)
