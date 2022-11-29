from __future__ import annotations

from functools import partial
from itertools import chain
from itertools import cycle
from itertools import islice
from itertools import repeat
from typing import Any
from typing import Dict

import numpy as np
import numpy.typing as npt
import shapely.geometry


def grouped(iterable, n):
    """s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."""
    return zip(*[iter(iterable)] * n)


# s -> (s0,s1), (s2,s3), (s3, s4), ...
pairwise = partial(grouped, n=2)


def swap(
    array: list[npt.NDArray[np.float32]],
    swap_pos: list[tuple[int, int]],
    zero_index: bool = False,
) -> list[npt.NDArray[np.float32]]:
    # in case of a single swap, swap_pos can be (pos1, pos2).
    # # Encapsulate the tuple in a list to have compatibility with general code
    # if not isinstance(swap_pos, list):
    #     swap_pos = [swap_pos]

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

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def nest_level(lst) -> int:
    if not isinstance(lst, list):
        return 0
    if not lst:
        return 1
    return max(nest_level(item) for item in lst) + 1


def flatten(items, seqtypes: tuple = (list, tuple)):
    try:
        for i, x in enumerate(items):
            while isinstance(x, seqtypes):
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


def split_mask(arr: npt.NDArray[Any], mask: list[bool] | npt.NDArray[bool]) -> list[npt.NDArray[Any]]:
    indices = np.nonzero(mask[1:] != mask[:-1])[0] + 1
    sp = np.split(arr, indices)
    sp = sp[0::2] if mask[0] else sp[1::2]
    return sp


def pad_infinite(iterable, padding=None):
    return chain(iterable, repeat(padding))


def pad(iterable, size, padding=None):
    return islice(pad_infinite(iterable, padding), size)


def almost_equals(polygon: shapely.geometry.Polygon, other: shapely.geometry.Polygon, tol: float = 1e-6) -> bool:
    return polygon.symmetric_difference(other).area < tol
