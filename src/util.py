import json
import os
from datetime import datetime


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
    return json.load(open(dict_file))

def get_time_as_string():
    '''Gets current time as string.'''
    now = datetime.now()
    date_time = now.strftime("%Y.%m.%d-%H.%M.%S")
    return date_time

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
