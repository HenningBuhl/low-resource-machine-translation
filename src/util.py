from datetime import datetime
from pytorch_lightning import seed_everything


import json
import os


CONST_BENCHMARKS_DIR = './benchmarks'
CONST_DATA_DIR = './data'
CONST_MODELS_DIR = './models'
CONST_RUNS_DIR = './runs'
CONST_TOKENIZERS_DIR = './tokenizers'


def create_dir(dir):
    '''Creates a directory if it does not already exists.'''
    if os.path.exists(dir):
        print(f'Dir "{dir}" already exists.')
    else:
        os.mkdir(dir)
        print(f'Dir "{dir}" does not exist, creating it.')

def create_dirs(*dirs):
    '''Creates all directories.'''
    for dir in dirs:
        create_dir(dir)

def save_dict(file, dict):
    '''Saves a dictionary to a file.'''
    with open(file, 'w') as f:
        json.dump(dict, f, indent=4)

def load_dict(dict_file):
    '''Loads a dictionary from a file.'''
    return DotDict(json.load(open(dict_file)))

def get_parallel_data_dir(base_dir, src_lang, tgt_lang):
    src_tgt_data = os.path.join(base_dir, f'{src_lang}-{tgt_lang}')
    tgt_src_data = os.path.join(base_dir, f'{tgt_lang}-{src_lang}')
    if os.path.exists(src_tgt_data):
        return src_tgt_data
    else:
        return tgt_src_data

def get_files(dir):
    '''Return all files that are present in a given directory.'''
    return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

def get_dirs(dir):
    '''Return all directories that are present in a given directory.'''
    return [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]

def get_time_as_string():
    '''Gets current time as string.'''
    now = datetime.now()
    date_time = now.strftime("%Y.%m.%d-%H.%M.%S")
    return date_time

def set_seed(seed):
    seed_everything(seed, workers=True)

def is_notebook():
    '''Checks whether the current environment is a jupyter notebook.'''
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif 'google.colab' in str(get_ipython()):
            return True   # Google Colab
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

# Dictionary which is capable of dot-notation.
class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            setattr(self, key, value)
        
    def __getattr__(self, name):
        return self[name]
