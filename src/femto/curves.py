from __future__ import annotations

import math

import numpy as np
from scipy import special
from scipy.interpolate import BPoly


def euler_bend(radius, theta, num_points):
    L = 2 * abs(radius) * theta  # total length of the Euler bend
    f = np.sqrt(np.pi * abs(radius) * L)  # Fresnel integral function are defined as function of (pi*t^2/2)

    t = np.linspace(0, L, num_points // 2)
    y, x = special.fresnel(t / f)
    z = np.zeros_like(x)
    return f * x, f * y, z


def sin_bridge(dy, dz, flat_peaks, radius, num_points):
    theta = np.arccos(1 - (np.abs(dy / 2) / radius))
    dx = 2 * abs(radius) * theta  # total length of the Euler bend
    x = np.linspace(0, dx, num_points // 2)

    tmp_cos = np.cos(np.pi / dx * x)
    y = 0.5 * dy * (1 - np.sqrt((1 + flat_peaks**2) / (1 + flat_peaks**2 * tmp_cos**2)) * tmp_cos)
    z = 0.5 * dz * (1 - np.cos(np.pi / dx * x))
    return x, y, z


def sin_sbend(dy, dz, flat_peaks, radius, num_points):
    return sin_bridge(dy=dy, dz=0, flat_peaks=flat_peaks, radius=radius, num_points=num_points)


def spline(
    dy,
    dz,
    radius,
    num_points,
    y_derivatives: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.0), (0.0, 0.0)),
    z_derivatives: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.0), (0.0, 0.0)),
):
    delta = np.sqrt(dy**2 + dz**2)
    theta = np.arccos(1 - (np.abs(delta / 2) / radius))
    dx = 2 * abs(radius) * theta  # total length of the Euler bend

    x = np.linspace(0, dx, num_points)
    y = BPoly.from_derivatives([0, dx], [[0, *y_derivatives[0]], [dy, *y_derivatives[-1]]])(x)
    z = BPoly.from_derivatives([0, dx], [[0, *z_derivatives[0]], [dz, *z_derivatives[-1]]])(x)
    return x, y, z


def poly_sbend(
    dy,
    dz,
    radius,
    num_points,
):
    return spline(
        dy=dy,
        dz=dz,
        radius=radius,
        num_points=num_points,
        y_derivatives=((0.0, 0.0), (0.0, 0.0)),
        z_derivatives=((0.0, 0.0), (0.0, 0.0)),
    )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x, y, z = euler_bend(radius=15, theta=math.radians(30), num_points=100)
    plt.figure(1)
    plt.clf()
    plt.plot(x, y)
    plt.show()

    x, y, z = spline(0.5, 0.0, radius=14, num_points=100)
    plt.figure(2)
    plt.clf()
    plt.plot(x, y)
    plt.show()
