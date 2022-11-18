import math

import pytest

from femto.PGMCompiler import PGMCompiler


@pytest.fixture
def param() -> dict:
    p = {
        "filename": "test.pgm",
        "export_dir": "G-Code",
        "samplesize": (25, 25),
        "rotation_angle": 1.0,
        "long_pause": 1.0,
        "short_pause": 0.025,
        "speed_pos": 10,
        "flip_x": True,
    }
    return p


@pytest.fixture
def empty_mk(param) -> PGMCompiler:
    return PGMCompiler(**param)


def test_default_values() -> None:
    G = PGMCompiler("prova")
    assert G.filename == "prova"
    assert G.export_dir == ""
    assert G.samplesize == (None, None)
    assert G.laser == "PHAROS"
    assert G.home is False
    assert G.new_origin == (0.0, 0.0)
    assert G.warp_flag is False
    assert G.n_glass == float(1.50)
    assert G.n_environment == float(1.33)
    assert G.rotation_angle == float(0.0)
    assert G.aerotech_angle == float(0.0)
    assert G.long_pause == float(0.5)
    assert G.short_pause == float(0.05)
    assert G.output_digits == int(6)
    assert G.speed_pos == float(5.0)
    assert G.flip_x is False
    assert G.flip_y is False


def test_gcode_values(param) -> None:
    G = PGMCompiler(**param)
    assert G.filename == "test.pgm"
    assert G.export_dir == "G-Code"
    assert G.samplesize == (25, 25)
    assert G.laser == "PHAROS"
    assert G.home is False
    assert G.new_origin == (0.0, 0.0)
    assert G.warp_flag is False
    assert G.n_glass == float(1.50)
    assert G.n_environment == float(1.33)
    assert G.rotation_angle == float(math.radians(1.0))
    assert G.aerotech_angle == float(0.0)
    assert G.long_pause == float(1.0)
    assert G.short_pause == float(0.025)
    assert G.output_digits == int(6)
    assert G.speed_pos == float(10)
    assert G.flip_x is True
    assert G.flip_y is False


def test_mk_from_dict(param) -> None:
    G = PGMCompiler.from_dict(param)
    assert G.filename == "test.pgm"
    assert G.export_dir == "G-Code"
    assert G.samplesize == (25, 25)
    assert G.laser == "PHAROS"
    assert G.home is False
    assert G.new_origin == (0.0, 0.0)
    assert G.warp_flag is False
    assert G.n_glass == float(1.50)
    assert G.n_environment == float(1.33)
    assert G.rotation_angle == float(math.radians(1.0))
    assert G.aerotech_angle == float(0.0)
    assert G.long_pause == float(1.0)
    assert G.short_pause == float(0.025)
    assert G.output_digits == int(6)
    assert G.speed_pos == float(10)
    assert G.flip_x is True
    assert G.flip_y is False
