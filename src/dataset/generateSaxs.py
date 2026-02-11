import numpy as np
import sys
from pathlib import Path
import h5py
import os

# internal
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from py_libraries.data.generation.saxs import UnitConvertor, main

if __name__ == '__main__':
    les_h5_path = r'D:\les_to_caracteristics\dataset\sphere_simulation_uniform_dielectric_c7-16_s1-1000_nt\les.h5'
    material = 'latex'
    shape = 'sphere'
    parameters_operator = 'stack'
    with h5py.File(les_h5_path, 'r') as f:
        mask = f['material'][:] == material.encode("utf-8")
        mask &= f['shape'][:] == shape.encode("utf-8")
        concentrations = f['concentration_part_cm_3'][:][mask]
        radius = f['width'][:][mask] / 2

    
    q = np.linspace(1e-3, 1, 300)

    env = "water"

    save_h5_filepath = rf'E:\autofill\data\saxs_{material}_{shape}.h5'
    
    max_workers=None
    other_attrs = {
        "author":"Esteban THEVENON",
        "type":"simulation"
    }
    
    # define parameters
    if shape == 'cube':
        parameters_operator = 'stack'
        shape = 'parallelepiped'
        other_attrs['shape'] = 'cube'
        length=np.arange(10, 101, 1)
        final_length = np.repeat(length, len(concentrations)) # [1, 2, 3] -> [1, 1, 2, 2, 3, 3,]
        parameters = {
            'concentration':np.array(list(concentrations) * len(length)), # [1, 2, 3] -> [1, 2, 3, 1, 2, 3]
            'length_a': UnitConvertor.nmToÅ(final_length),   # height
            'length_b': UnitConvertor.nmToÅ(final_length), # width
            'length_c': UnitConvertor.nmToÅ(final_length)  # length
            }
    elif shape == 'parallelepiped':
        parameters = {
            'concentration':concentrations,
            'length_a': UnitConvertor.nmToÅ(np.arange(10, 51, 5)),   # height
            'length_b': UnitConvertor.nmToÅ(np.arange(20, 101, 10)), # width
            'length_c': UnitConvertor.nmToÅ(np.arange(50, 201, 10))  # length
            }
    elif shape == 'sphere':
        parameters = {
            'concentration':concentrations,
            'radius': UnitConvertor.nmToÅ(radius)
            }
    elif shape == 'cylinder':
        parameters = {
        'concentration':concentrations,
            'radius': UnitConvertor.nmToÅ(np.arange(10, 101, 20)/2),
            'length': UnitConvertor.nmToÅ(np.arange(5, 16, 4)),
            }
    else:
        raise Exception(f'{shape} parameters is not define. Please set this shape before going farwer.')
    
    main(
        q = q,
        parameters = parameters,
        parameters_operator = parameters_operator,
        shape = shape,
        material = material,
        env=env,
        other_attrs=other_attrs,
        save_h5_filepath=save_h5_filepath,
        max_workers=max_workers,
    )