from typing import List
from femto import PGMCompiler
from collections.abc import Iterable
from itertools import zip_longest
import os
import numpy as np


def export_array(gc: PGMCompiler,
                 points: np.ndarray,
                 f_array: List = []):

    if points.shape[-1] == 2:
        x_array, y_array = np.matmul(points, gc._t_matrix(dim=2)).T
        z_array = [None]
    else:
        x_array, y_array, z_array = np.matmul(points, gc._t_matrix()).T

    if not isinstance(f_array, Iterable):
        f_array = [f_array]

    instructions = [gc._format_args(x, y, z, f)
                    for (x, y, z, f) in zip_longest(x_array,
                                                    y_array,
                                                    z_array,
                                                    f_array)]
    gc._instructions = [f'LINEAR {line}\n' for line in instructions]


def make_trench(gc: PGMCompiler,
                col: List,
                col_index: int,
                base_folder,
                dirname: str = 's-trench',
                u: List = None,
                nboxz: int = 4,
                hbox: float = 0.075,
                zoff: float = 0.020,
                deltaz: float = 0.0015,
                tspeed: float = 4,
                angle: float = 0.0,
                ind_rif: float = 1.5/1.33,
                speed_pos: float = 5,
                pause: float = 0.5):

    trench_directory = os.path.join(dirname,
                                    f'trenchCol{col_index+1:03}')
    col_dir = os.path.join(os.getcwd(), trench_directory)
    os.makedirs(col_dir, exist_ok=True)

    # Export paths
    for i, trench in enumerate(col.trench_list):
        wall_filename = os.path.join(col_dir, f'trench{i+1:03}_wall')
        floor_filename = os.path.join(col_dir, f'trench{i+1:03}_floor')

        # Export wall
        t_gc = PGMCompiler(wall_filename, ind_rif=ind_rif, angle=angle)
        export_array(t_gc, np.stack((trench.border), axis=-1), f_array=tspeed)
        t_gc.close()
        del t_gc

        # Export floor
        t_gc = PGMCompiler(floor_filename, ind_rif=ind_rif, angle=angle)
        export_array(t_gc, np.stack((trench.floor), axis=-1), f_array=tspeed)
        t_gc.close()
        del t_gc

    gc.dvar(['ZCURR'])

    for nbox in range(nboxz):
        for t_index, trench in enumerate(col.trench_list):
            # load filenames (wall/floor)
            wall_filename = f'trench{t_index+1:03}_wall.pgm'
            floor_filename = f'trench{t_index+1:03}_floor.pgm'
            wall_path = os.path.join(base_folder,
                                     trench_directory,
                                     wall_filename)
            floor_path = os.path.join(base_folder,
                                      trench_directory,
                                      floor_filename)

            x0, y0 = trench.border[:, 0]
            z0 = (nbox*hbox - zoff)/gc.ind_rif
            gc.comment(f'+--- TRENCH #{t_index+1}, LEVEL {nbox+1} ---+')
            gc.load_program(wall_path)
            gc.load_program(floor_path)
            gc.shutter('OFF')
            gc.move_to([x0, y0, z0], speed_pos=speed_pos)

            gc.instruction(f'$ZCURR = {z0:.6f}')
            gc.shutter('ON')
            gc.repeat(int(np.ceil((hbox+zoff)/deltaz)))
            gc.farcall(wall_filename)
            gc.instruction(f'$ZCURR = $ZCURR + {deltaz/gc.ind_rif:.6f}')
            gc.instruction('LINEAR Z$ZCURR')
            gc.end_repeat()

            if u is not None:
                gc.instruction(f'LINEAR U{u[-1]:.6f}')
            gc.dwell(pause)
            gc.farcall(floor_filename)
            gc.shutter('OFF')
            if u is not None:
                gc.instruction(f'LINEAR U{u[0]:.6f}')

            gc.remove_program(wall_path)
            gc.remove_program(floor_path)
