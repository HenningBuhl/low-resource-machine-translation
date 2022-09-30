from util import *


CONST_DATA_DIR = './data'
CONST_MODELS_DIR = './models'
CONST_RUNS_DIR = './runs'
CONST_TOKENIZERS_DIR = './tokenizers'


class ExperimentPathManager():
    '''
    A class managing all paths required for an experiment. One experiment consist of training and manging
    one or more models.
    '''

    def __init__(self, run_name, *models_names, run_dir=None):
        self.run_name = run_name
        self.models_names = models_names

        # Use a new run_dir if none was passed.
        if run_dir is None:
            self.run_dir = os.path.join(CONST_RUNS_DIR, f'{self.run_name}-{get_time_as_string()}')
        else:
            self.run_dir = run_dir

        # Args file.
        self.args_file = os.path.join(self.run_dir, 'args.json')

        # Model path managers.
        for models_name in self.models_names:
            mpm = ModelPathManager(os.path.join(self.run_dir, models_name))
            setattr(self, models_name, mpm)

    def init(self):
        '''Creates all required directories.'''

        # Constant directories.
        create_dirs(CONST_DATA_DIR, CONST_MODELS_DIR, CONST_RUNS_DIR, CONST_TOKENIZERS_DIR)

        # Create run_dir.
        create_dir(self.run_dir)

        # Model path managers.
        for models_name in self.models_names:
            mpm = getattr(self, models_name)
            mpm.init()


class ModelPathManager():
    '''A class managing all paths one model requires.'''
    def __init__(self, model_dir):
        self.model_dir = model_dir

        # Directories.
        self.checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')
        self.metrics_dir = os.path.join(self.model_dir, 'metrics')

        # Files.
        self.untrained_model_file = os.path.join(self.model_dir, 'untrained_model.pt')
        self.model_file = os.path.join(self.model_dir, 'final_model.pt')
        self.metrics_file = os.path.join(self.metrics_dir, 'metrics.json')
        self.metrics_svg_template = os.path.join(self.metrics_dir, '{}.svg')

    def init(self):
        '''Creates all required directories.'''

        create_dir(self.model_dir)
        create_dir(self.checkpoint_dir)
        create_dir(self.metrics_dir)


def get_parallel_data_dir(src_lang, tgt_lang):
    src_tgt_data = os.path.join(CONST_DATA_DIR, f'{src_lang}-{tgt_lang}')
    tgt_src_data = os.path.join(CONST_DATA_DIR, f'{tgt_lang}-{src_lang}')
    if os.path.exists(src_tgt_data):
        return src_tgt_data
    else:
        return tgt_src_data

def get_files(dir):
    '''Return all files that are present in a given directory.'''
    return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
