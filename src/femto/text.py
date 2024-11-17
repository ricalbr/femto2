from __future__ import annotations

import attr
import svgpathtools
import shapely
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Any
import attrs
import shapely.geometry
import shapely.ops
import shapely.affinity
from femto import logger

from femto.utils import _fonts
from femto.helpers import normalize_phase

# Define array type
nparray = npt.NDArray[np.float32]


@attrs.define(kw_only=True)
class Alignment:
    """
    Alignment helper class.

    Given a bounding box in the form of ``((min_x, min_y), (max_x, max_y))``, this class
    returns special points of this box::

        (max_y)    top +------+------+
                       |      |      |
                       |      |      |
               center  +------+------+
                       |      |      |
                       |      |      |
        (min_y) bottom +------+------+
                  (min_x)         (max_x)
                    Left   Center   Right

    Alignment options are given as ``-`` separated tuple, allowing for combinations of
    ``left``, ``center``, ``right`` with ``bottom``, ``center``, ``top``.
    """

    alignment: str = attrs.field(validator=attrs.validators.instance_of(str), default='left-bottom')
    _ALIGNMENT: dict[str] = {
        "x": {
            "left": lambda coord: coord[0][0],
            "center": lambda coord: np.mean(coord[:, 0]),
            "right": lambda coord: coord[1][0],
        },
        "y": {
            "bottom": lambda coord: coord[0][1],
            "center": lambda coord: np.mean(coord[:, 1]),
            "top": lambda coord: coord[1][1],
        },
    }

    @alignment.validator
    def _alignment_validator(self, attribute, value) -> None:
        """Validator for the alignment attribute."""

        options = [option.strip() for option in value.split("-")]
        if len(options) != 2:
            logger.error('Alignment option string must be two options separated by a dash.')
            raise AttributeError('Alignment option string must be two options separated by a dash.')
        if options[0] not in self._ALIGNMENT["x"]:
            logger.error(f'x-axis alignment option must be one of {self._ALIGNMENT["x"].keys()}')
            raise AttributeError(f'x-axis alignment option must be one of {self._ALIGNMENT["x"].keys()}')
        if options[1] not in self._ALIGNMENT["y"]:
            logger.error(f'y-axis alignment option must be one of {self._ALIGNMENT["y"].keys()}')
            raise AttributeError(f'y-axis alignment option must be one of {self._ALIGNMENT["y"].keys()}')

    @property
    def alignment_functions(self):
        """Returns a 2-tuple of functions, calculating the offset coordinates for a given bounding box.

        Returns
        -------
        tuple(str, str)
            Tuple of functions.
        """
        options = self.alignment.split("-")
        return self._ALIGNMENT["x"][options[0]], self._ALIGNMENT["y"][options[1]]

    def calculate_offset(self, bbox):
        """Calculate the coordinates of the current alignment for the given bounding box *bbox*.

        Parameters
        ----------
        bbox: tuple(float, float)
            Bounding box in the ``((min_x, min_y), (max_x, max_y))`` format.

        Returns
        -------
        tuple(float, float)
            (x, y) offset coordinates.
        """
        bbox = np.asarray(bbox)
        alignment_fun = self.alignment_functions
        offset = np.array([-alignment_fun[i](bbox) for i in [0, 1]])
        return offset


@attrs.define(kw_only=True, repr=False, init=False)
class Text:
    text: str = ""
    alignment_position: str = "left-bottom"
    origin: list[float] | nparray = [1.0, 1.0]
    height: float = 1.0
    angle: float = 0.0
    font: str = "stencil"
    line_spacing: float = 1.5
    true_bbox_alignment: bool = False

    _alignment: Alignment = attrs.field(factory=Alignment)
    _bbox: tuple[float, float] | None = None
    _shapely_object: shapely.geometry.MultiPolygon | None = None  #: Multipolygon shape of the text.
    _letters: list[shapely.geometry.Polygon] = attr.field(factory=list)

    def __init__(self, **kwargs: Any):
        filtered: dict[str, Any] = {att.name: kwargs[att.name] for att in self.__attrs_attrs__ if att.name in kwargs}
        self.__attrs_init__(**filtered)

    def __attrs_post_init__(self) -> None:
        if np.array(self.origin).shape!= (2,):
            logger.error('Origin not valid. Need an iterable of two elements.')
            raise AttributeError('Origin not valid. Need an iterable of two elements.')
        self.origin = np.array(self.origin)

        if self.height < 0:
            logger.error(f'Height must be a positive number. Given {self.height}.')
            raise AttributeError(f'Height must be a positive number. Given {self.height}.')

        if self.font not in _fonts.FONTS:
            logger.error(f'Font is {self.font} unknown, must be one of {_fonts.FONTS.keys()}')
            raise AttributeError(f'Font is {self.font} unknown, must be one of {_fonts.FONTS.keys()}')

        self.angle = normalize_phase(self.angle)
        self.line_spacing = self.height * self.line_spacing
        _alignment = Alignment(alignment=self.alignment_position)
        _bbox = None
        _shapely_object = None

    def _invalidate(self) -> None:
        self._bbox = None
        self._shapely_object = None

    @property
    def alignment(self) -> str:
        return self._alignment.alignment

    @alignment.setter
    def alignment(self, alignment_pos) -> None:
        self._invalidate()
        self._alignment = Alignment(alignment=alignment_pos)

    @property
    def bounding_box(self) -> tuple[float, float]:
        if self._bbox is None:
            self.get_shapely_object()
        return self._bbox

    @property
    def letters(self) -> list[shapely.geometry.Polygon]:
        if not self._letters:
            self.get_shapely_object()
        return self._letters

    def get_shapely_object(self):
        if not self.text:
            self._bbox = None
            return shapely.geometry.Polygon()

        if self._shapely_object:
            return self._shapely_object

        # Let's do the actual rendering
        polygons = list()

        special_handling_chars = "\n"
        font = _fonts.FONTS[self.font]

        # Check the text
        for char in self.text:
            assert char not in special_handling_chars or char not in font, (
                f'Character "{char}" is not supported by ' f'font "{self.font}"'
            )

        max_x = 0
        cursor_x, cursor_y = 0, 0
        for i, char in enumerate(self.text):
            if char == "\n":
                cursor_x, cursor_y = 0, cursor_y - self.line_spacing
                continue

            char_font = font[char]
            cursor_x += char_font["width"] / 2 * self.height

            for line in char_font["lines"]:
                points = np.array(line).T * self.height + (cursor_x, cursor_y)
                polygons.append(shapely.geometry.Polygon(points))

            # Add kerning (space between letters)
            if i < len(self.text) - 1 and self.text[i + 1] not in special_handling_chars:
                kerning = char_font["kerning"][self.text[i + 1]]
                cursor_x += (char_font["width"] / 2 + kerning) * self.height

            max_x = max(max_x, cursor_x + char_font["width"] / 2 * self.height)

        merged_polygon = shapely.ops.unary_union(polygons)

        # Handle the alignment, translation and rotation
        if not self.true_bbox_alignment:
            bbox = np.array([[0, max_x], [cursor_y, self.height]]).T
        else:
            bbox = np.array(merged_polygon.bounds).reshape(2, 2)

        offset = self._alignment.calculate_offset(bbox)

        if not np.isclose(normalize_phase(self.angle), 0):
            aligned_text = shapely.affinity.translate(merged_polygon, *offset)
            rotated_text = shapely.affinity.rotate(aligned_text, self.angle, origin=[0, 0], use_radians=True)
            final_text = shapely.affinity.translate(rotated_text, *self.origin)
        else:
            final_text = shapely.affinity.translate(merged_polygon, *(offset + self.origin))

        # Convert single letter text to Multipolygon to read the letters using the .geoms property
        if isinstance(final_text, shapely.geometry.Polygon):
            final_text = shapely.geometry.MultiPolygon([final_text])

        self._bbox = np.array(final_text.bounds).reshape(2, 2)
        self._shapely_object = final_text
        self._letters.extend(list(final_text.geoms))
        return final_text


def _example():
    text = Text(origin=[0, 0], height=0.5, text="c", alignment="left-bottom")

    fig, ax = plt.subplots()
    print(list(text.get_shapely_object().geoms))
    for polygon in text.get_shapely_object().geoms:
        # polygon = polygon.buffer(-text.height * 0.025, single_sided=True)
        x, y = polygon.exterior.xy
        ax.fill(x, y, alpha=0.5, fc="r", edgecolor="black", linewidth=1)
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    _example()
