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


def sin_bend(dx, dy, dz, num_points, flat_peaks=0):

    x = np.linspace(0, dx, num_points)
    tmp_cos = np.cos(np.pi / dx * x)
    y = 0.5 * dy * (1 - np.sqrt((1 + flat_peaks**2) / (1 + flat_peaks**2 * tmp_cos**2)) * tmp_cos)
    z = 0.5 * dz * (1 - np.cos(np.pi / dx * x))
    return x, y, z


def spline(
    dx,
    dy,
    dz,
    num_points,
    y_derivatives: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.0), (0.0, 0.0)),
    z_derivatives: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.0), (0.0, 0.0)),
):
    x = np.linspace(0, dx, num_points)
    y = BPoly.from_derivatives([0, dx], [[0, *y_derivatives[0]], [dy, *y_derivatives[-1]]])(x)
    z = BPoly.from_derivatives([0, dx], [[0, *z_derivatives[0]], [dz, *z_derivatives[-1]]])(x)
    return x, y, z


def spline_bridge(
    dx,
    dy,
    dz,
    num_points,
):
    xi, yi, zi = spline(
        dx=dx / 2,
        dy=dy / 2,
        dz=dz,
        num_points=num_points // 2,
        y_derivatives=((0.0, 0.0), (dy / dx, 0.0)),
        z_derivatives=((0.0, 0.0), (0.0, 0.0)),
    )
    xf, yf, zf = spline(
        dx=dx / 2,
        dy=dy / 2,
        dz=-dz,
        num_points=num_points // 2,
        y_derivatives=((dy / dx, 0.0), (0.0, 0.0)),
        z_derivatives=((0.0, 0.0), (0.0, 0.0)),
    )
    return np.append(xi, xf + xi[-1]), np.append(yi, yf + yi[-1]), np.append(zi, zf + zi[-1])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x, y, z = euler_bend(radius=15, theta=math.radians(30), num_points=100)
    plt.figure(1)
    plt.clf()
    plt.plot(x, y)
    # plt.show()

    x, y, z = spline(0.5, 0.3, dz=0.0, num_points=100)
    plt.figure(2)
    plt.clf()
    plt.plot(x, y)
    # plt.show()

    x, y, z = sin_bend(dx=0.5, dy=0.6, dz=0.0, num_points=100, flat_peaks=1)
    plt.figure(3)
    plt.clf()
    plt.plot(x, y)
    plt.show()

    x, y, z = spline_bridge(dx=5, dy=0.06, dz=0.03, num_points=100)
    plt.figure(3)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x, y, z, label='parametric curve')
    ax.legend()

    plt.show()
