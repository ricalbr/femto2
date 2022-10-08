# Author and license
__author__ = "Riccardo Albiero"
__license__ = "MIT"

# Optional variables
__description__ = 'Python library for design and fabrication of femtosecond laser written integrated photonic circuits.'
__keywords__ = 'python g-code integrated-circuits quantum-optics'

# Support for "from femto import *"
# @see: https://stackoverflow.com/a/41895257
# @see: https://stackoverflow.com/a/35710527
__all__ = [
    'Parameters',
    'WaveguideParameters',
    'TrenchParameters',
    'GcodeParameters',
    'LaserPath',
    'Waveguide',
    'Marker',
    'Trench',
    'TrenchColumn',
    'PGMCompiler',
    'PGMTrench',
    'Device',
    'Cell',
    'helpers',
]

from .helpers import *
from .Parameters import WaveguideParameters, TrenchParameters, GcodeParameters
from .LaserPath import LaserPath
from .Waveguide import Waveguide, coupler
from .Marker import Marker
from .Trench import Trench, TrenchColumn
from .PGMCompiler import PGMCompiler, PGMTrench
from .Cell import Cell, Device
