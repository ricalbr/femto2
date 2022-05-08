# -*- coding: utf-8 -*-
"""
Created on Tue May  3 23:37:52 2022

@author: enric
"""

from femto.objects import Waveguide
from femto.compiler import PGMCompiler
import matplotlib.pyplot as plt
import paramFile as p


cell = {
    'waveguide': [Waveguide() for _ in range(36)]
}

[xi, yi, zi] = [p.x0, p.y0, p.depth]
for h in range(3):
    for i in range(3):
        'GD'
        cell['waveguide'][4*i+12*h].num_scan = 4+2*h
        cell['waveguide'][4*i+12*h].start([xi, 0.4*i+yi+1.2*h, zi])
        cell['waveguide'][4*i+12*h].linear([p.xlen, 0, 0], speed=2*i+p.speed)
        cell['waveguide'][4*i+12*h].end()

        'SM20X16'
        cell['waveguide'][4*i+1+12*h].num_scan = 4+2*h
        cell['waveguide'][4*i+1+12*h].start([xi, 0.4*i+yi+0.08+1.2*h, zi])
        cell['waveguide'][4*i+1+12*h].linear(p.increment, speed=2*i+p.speed)
        cell['waveguide'][4*i+1+12*h].sin_bend(p.d1, p.radius, speed=2*i+p.speed, N=p.N)
        for k in range(8):
            cell['waveguide'][4*i+1+12*h].sin_bend(-p.d1, p.radius, speed=2*i+p.speed, N=p.N)
            cell['waveguide'][4*i+1+12*h].sin_bend(p.d1, p.radius, speed=2*i+p.speed, N=p.N)
            cell['waveguide'][4*i+1+12*h].sin_bend(-p.d2, p.radius, speed=2*i+p.speed, N=p.N)
            cell['waveguide'][4*i+1+12*h].sin_bend(p.d1, p.radius, speed=2*i+p.speed, N=p.N)
            cell['waveguide'][4*i+1+12*h].sin_bend(-p.d1, p.radius, speed=2*i+p.speed, N=p.N)
            cell['waveguide'][4*i+1+12*h].sin_bend(p.d2, p.radius, speed=2*i+p.speed, N=p.N)
        cell['waveguide'][4*i+1+12*h].sin_bend(-p.d1, p.radius, speed=2*i+p.speed, N=p.N)
        cell['waveguide'][4*i+1+12*h].linear([16, 0, 0], speed=2*i+p.speed)
        cell['waveguide'][4*i+1+12*h].end()

        'ACC20x16'
        cell['waveguide'][4*i+2+12*h].num_scan = 4+2*h
        cell['waveguide'][4*i+2+12*h].start([xi, 0.4*i+yi+0.16+1.2*h, zi])
        cell['waveguide'][4*i+2+12*h].linear(p.increment, speed=2*i+p.speed)
        cell['waveguide'][4*i+2+12*h].sin_bend(p.d1, p.radius, speed=2*i+p.speed, N=p.N)
        for k in range(8):
            cell['waveguide'][4*i+2+12*h].sin_bend(-p.d1, p.radius, speed=2*i+p.speed, N=p.N)
            cell['waveguide'][4*i+2+12*h].sin_bend(p.d1, p.radius, speed=2*i+p.speed, N=p.N)
            cell['waveguide'][4*i+2+12*h].sin_bend(-p.d2, p.radius, speed=2*i+p.speed, N=p.N)
            cell['waveguide'][4*i+2+12*h].sin_bend(p.d1, p.radius, speed=2*i+p.speed, N=p.N)
            cell['waveguide'][4*i+2+12*h].sin_bend(-p.d1, p.radius, speed=2*i+p.speed, N=p.N)
            cell['waveguide'][4*i+2+12*h].sin_bend(p.d2, p.radius, speed=2*i+p.speed, N=p.N)
        cell['waveguide'][4*i+2+12*h].sin_bend(-p.d1, p.radius, speed=2*i+p.speed, N=p.N)
        cell['waveguide'][4*i+2+12*h].linear([16, 0, 0], speed=2*i+p.speed)
        cell['waveguide'][4*i+2+12*h].end()

        cell['waveguide'][4*i+3+12*h].num_scan = 4+2*h
        cell['waveguide'][4*i+3+12*h].start([xi, 0.4*i+yi+0.24+1.2*h, zi])
        cell['waveguide'][4*i+3+12*h].linear(p.increment, speed=2*i+p.speed)
        cell['waveguide'][4*i+3+12*h].sin_bend(-p.d1, p.radius, speed=2*i+p.speed, N=p.N)
        for k in range(8):
            cell['waveguide'][4*i+3+12*h].sin_bend(p.d1, p.radius, speed=2*i+p.speed, N=p.N)
            cell['waveguide'][4*i+3+12*h].sin_bend(-p.d1, p.radius, speed=2*i+p.speed, N=p.N)
            cell['waveguide'][4*i+3+12*h].sin_bend(p.d2, p.radius, speed=2*i+p.speed, N=p.N)
            cell['waveguide'][4*i+3+12*h].sin_bend(-p.d1, p.radius, speed=2*i+p.speed, N=p.N)
            cell['waveguide'][4*i+3+12*h].sin_bend(p.d1, p.radius, speed=2*i+p.speed, N=p.N)
            cell['waveguide'][4*i+3+12*h].sin_bend(-p.d2, p.radius, speed=2*i+p.speed, N=p.N)
        cell['waveguide'][4*i+3+12*h].sin_bend(p.d1, p.radius, speed=2*i+p.speed, N=p.N)
        cell['waveguide'][4*i+3+12*h].linear([16, 0, 0], speed=2*i+p.speed)
        cell['waveguide'][4*i+3+12*h].end()


# Plot
fig, ax = plt.subplots()
for wg in cell['waveguide']:
    ax.plot(wg.x[:-1], wg.y[:-1], '-b', linewidth=2.5)
    ax.plot(wg.x[-2:], wg.y[-2:], ':k', linewidth=1)


gc = PGMCompiler('opt_acc.pgm', ind_rif=p.ind_env, angle=p.angle)
gc.header()
for k in range(3):
    gc.rpt(cell['waveguide'][k*13].num_scan)
    for i, wg in enumerate(cell['waveguide'][0+12*k:12+12*k]):
        print(i)
        gc.comment(f' +--- Modo: {i+1}---+')
        gc.point_to_instruction(wg.M)
    gc.endrpt()
gc.compile_pgm()
