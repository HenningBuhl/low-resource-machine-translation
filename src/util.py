import json
import os
from datetime import datetime


# Create a directory only if it does not already exists.
def create_dir(dir):
    if os.path.exists(dir):
        print(f'Dir "{dir}" already exists.')
    else:
        os.mkdir(dir)
        print(f'Dir "{dir}" does not exist, creating it.')

# Save a python dictionary.
def save_dict(dir, dictionary, name):
    f = open(os.path.join(dir, f'{name}.json'), 'w')
    json.dump(dictionary, f)
    f.close()

# Read a python dictionary.
def read_dict(dict_file):
    return json.load(open(dict_file))

# Dictionary which is capable of dot-notation.
class dotdict(dict):
    def __init__(self, *args, **kwargs):
        super(dotdict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            setattr(self, key, value)
        
    def __getattr__(self, name):
        return self[name]

# Iterate over batches.
def batch(iterable, batch_size=1):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]

# Get current time as string.
def get_time_as_string():
    now = datetime.now()
    date_time = now.strftime("%Y.%m.%d-%H.%M.%S")
    return date_time

# Check checkter the current environment is a jupyter notebook.
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
