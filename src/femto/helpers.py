from __future__ import annotations

from functools import partial
from itertools import chain
from itertools import cycle
from itertools import islice
from itertools import repeat
from typing import Any
from typing import Dict
from typing import Iterable

import numpy as np
import numpy.typing as npt
import shapely.geometry
from numpy import generic


def grouped(iterable: Iterable[Any], n: int) -> Iterable[Any]:
    """
    Gruoup an iterable in sub-groups of n elements.

    s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ...

    :param iterable: iterable to group
    :param n: size of the sub-groups
    :return:
    """
    return zip(*[iter(iterable)] * n)


# s -> (s0,s1), (s2,s3), (s3, s4), ...
pairwise = partial(grouped, n=2)


def swap(
    array: list[Any],
    swap_pos: list[tuple[int, int]],
    zero_index: bool = False,
) -> list[Any]:
    """
    Swaps elements

    :param array:
    :param swap_pos:
    :param zero_index:
    :return:
    """
    # in case of a single swap, swap_pos can be (pos1, pos2).

    for pos1, pos2 in swap_pos:
        if zero_index is False:
            pos1 -= 1
            pos2 -= 1
        array[pos1], array[pos2] = array[pos2], array[pos1]
    return array


def listcast(x):
    # cast any obj to a list
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]


class Dotdict(Dict[Any, Any]):
    """dot.notation access to dictionary attributes"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super().__delitem__(key)
        del self.__dict__[key]


def nest_level(lst) -> int:
    if not isinstance(lst, list):
        return 0
    if not lst:
        return 1
    return max(nest_level(item) for item in lst) + 1


def flatten(items):
    try:
        for i, x in enumerate(items):
            while isinstance(x, (list, tuple)) and not isinstance(x, (str, bytes)):
                items[i : i + 1] = x
                x = items[i]
    except IndexError:
        pass
    return items


def sign():
    return cycle([1, -1])


def unique_filter(arrays: list[npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
    # data matrix
    data = np.stack(arrays, axis=-1).astype(np.float32)

    # empty array list
    if data.size == 0:
        return np.array([])

    # mask
    mask = np.diff(data, axis=0)
    mask = np.sum(np.abs(mask), axis=1, dtype=bool)
    mask = np.insert(mask, 0, True)

    # filtered data
    return np.array(data[mask])


def split_mask(arr: npt.NDArray[Any], mask: list[bool] | npt.NDArray[generic]) -> list[npt.NDArray[Any]]:
    indices = np.nonzero(mask[1:] != mask[:-1])[0] + 1
    sp = np.split(arr, indices)
    sp = sp[0::2] if mask[0] else sp[1::2]
    return sp


def pad_infinite(iterable: Iterable[Any], padding: Any = None):
    return chain(iterable, repeat(padding))


def pad(iterable, size, padding=None):
    return islice(pad_infinite(iterable, padding), size)


def almost_equals(
    polygon: shapely.geometry.polygon.Polygon,
    other: shapely.geometry.polygon.Polygon,
    tol: float = 1e-6,
) -> bool:
    return bool(polygon.symmetric_difference(other).area < tol)
