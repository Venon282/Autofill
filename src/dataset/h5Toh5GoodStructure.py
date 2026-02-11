from enum import Enum
import sys
import h5py
import os
import numpy as np
from pathlib import Path
import operator as op

# internal
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from py_libraries.other.loggingUtils import getLogger
from py_libraries.constant.Constant import Constants
from py_libraries.file.h5 import append as h5Append

logger = getLogger(__name__)
Constants.LES_COLS = ['concentration', 'csv_index', 'data_wavelength', 'data_y', 'diameter_nm', 'length_nm']
Constants.SAXS_COLS = ['concentration', 'csv_index', 'data_q', 'data_y', 'diameter_nm', 'length_nm']
Constants.ATTRS = ['material', 'technique', 'shape']

class Type(Enum):
    SAXS=0
    LES=1

def _process(
    h5_path:str,
    new_h5_path:str,
    corresponding_cols:dict,
    corresponding_attrs:dict,
    ):
    def attrsAppend(f, col, values):
        f.attrs[col] = values
        
    def _processCorresponding(corresponding, f_src, f_dest, appendFunction):
        for dest_col, element in corresponding.items():
            if isinstance(element, str):
                appendFunction(f_dest, dest_col, f_src[element][:])
            elif element is None:
                appendFunction(f_dest, dest_col, np.arange(len(f_src[next(iter(f_src.keys()))])))
            elif isinstance(element, dict):
                froms_, values_ = element['from'], element['value']
                is_froms_list, is_values_list = isinstance(froms_, list), isinstance(values_, list)
                
                if is_froms_list and is_values_list:
                    if len(froms_) != len(values_):
                        raise Exception(f'For {dest_col}, from and value size mismatch {len(froms_)} != {len(values_)}')
                    
                elif is_froms_list and not is_values_list:
                    values_ = [values_ for _ in range(len(froms_))]
                elif not is_froms_list and is_values_list:
                    froms_ = [froms_ for _ in range(len(values_))]
                else:
                    values_ = [values_]
                    froms_ = [froms_]
                    
                for from_, value_ in zip(froms_, values_):
                    if from_ == 'col':
                        data = f_src[value_][:]
                    elif from_.startswith('op_'):
                        data = getattr(op, from_.split('_')[1])(data, value_)
                    elif from_ == 'attrs':
                        data = f_src.attrs[value_]
                
                appendFunction(f_dest, dest_col, data)

            else:
                raise ValueError('Each value corresponding must be either a dict or a str.')
            
    with h5py.File(h5_path, 'r') as f, \
         h5py.File(new_h5_path, 'w') as new_f:
        _processCorresponding(corresponding_cols, f_src=f, f_dest=new_f, appendFunction=h5Append)
        _processCorresponding(corresponding_attrs, f_src=f, f_dest=new_f, appendFunction=attrsAppend)

def verifCorresponding(user, ask):
    if len(set(user) ^ set(ask)) > 0: # elements in user and ask that is not in common
        raise Exception('User keys must correspond strictly to the desire ones')

def process(
            file_type:Type, 
            h5_path:str,
            new_h5_path:str,
            corresponding_cols:dict,
            corresponding_attrs:dict,
            overwrite:bool=False,
            ):
    if not isinstance(file_type, Type):
        raise AttributeError('file_type must be an instance of Type class.')
    
    if not os.path.exists(h5_path):
        raise AttributeError(f'The h5_path do not exist: {h5_path}')
    
    if h5_path == new_h5_path:
        raise AttributeError('h5_path and new_h5_path can\'t be the same.')
    
    if os.path.splitext(h5_path)[1] not in ('.h5', '.hdf5'):
        raise AttributeError(f'The h5_path must be an h5 file and not a {os.path.splitext(h5_path)[1]}. [.h5, .hdf5]')
    
    if os.path.splitext(new_h5_path)[1] not in ('.h5', '.hdf5'):
        raise AttributeError(f'The new_h5_path must be an h5 file and not a {os.path.splitext(new_h5_path)[1]}. [.h5, .hdf5]')
    
    if os.path.exists(new_h5_path):
        if overwrite:
            logger.warning('The new_h5_path will be overwrite as it already exists and overwrite is set to True.')
        else:
            raise AttributeError('new_h5_path already exist. If you set overwrite to True, you can overwrite it.')
        
    verifCorresponding(corresponding_cols.keys(), Constants.LES_COLS if file_type == Type.LES else Constants.SAXS_COLS if file_type == Type.SAXS else (_ for _ in ()).throw(Exception(f'Unundel type {file_type.name}')))
    verifCorresponding(corresponding_attrs.keys(), Constants.ATTRS)
    
    # Get the function arguments before set anythings    
    arguments = locals()
    
    del arguments['file_type']
    del arguments['overwrite']
    _process(**arguments)

if __name__ == '__main__':
    file_type = Type.LES
    h5_path = r'E:\signals\sphere_les_ag_new_meta.h5'
    #h5_path = r'E:\signals\saxs_au_sphere.h5'
    # new_h5_path = r'E:\signals\sphere_les_ag_new_meta2.h5'
    new_h5_path = r'C:\Users\ET281306\Downloads\delete_me.h5'
    overwrite = True
    
    with h5py.File(h5_path, 'r') as f:
        print(f.keys())
        print(f.attrs.keys())
    
    les_corresponding_cols = {
       'csv_index':'csv_index',
       'concentration':'concentration',
       'data_wavelength':'data_wavelength',
       'data_y':'data_y',
       'diameter_nm':'Diameter_nm',
       'length_nm':'Diameter_nm'
    }
    
    saxs_corresponding_cols = {
       'csv_index':None,
       'concentration':'concentration',
       'data_q':'q',
       'data_y':'intensities',
       'diameter_nm':{
            'from':['col', 'op_mul'],
            'value':['radius', 2]
       },
       'length_nm':{
            'from':['col', 'op_mul'],
            'value':['radius', 2]
       }
    }
    
    corresponding_cols = les_corresponding_cols if file_type == Type.LES else saxs_corresponding_cols if file_type == Type.SAXS else (_ for _ in ()).throw(Exception(''))
    
    corresponding_attrs = {
        'material':{
            'from':'attrs',
            'value':'material'
        }, 
        'technique':{
            'from':'attrs',
            'value':'technique'
        }, 
        'shape':{
            'from':'attrs',
            'value':'shape'
        }
    }
    
    process(
        file_type=file_type,
        h5_path=h5_path,
        new_h5_path=new_h5_path,
        corresponding_cols=corresponding_cols,
        corresponding_attrs=corresponding_attrs,
        overwrite=overwrite
    )