from util import *


CONST_DATA_DIR = './data'
CONST_MODELS_DIR = './models'
CONST_RUNS_DIR = './runs'
CONST_TOKENIZERS_DIR = './tokenizers'


class PathManager():
    def __init__(self, run_name, *models_names, run_dir=None):
        self.run_name = run_name
        self.models_names = models_names

        # Run dir.
        if run_dir is None:
            self.run_dir = os.path.join(CONST_RUNS_DIR, f'{self.run_name}-{get_time_as_string()}')
        else:
            self.run_dir = run_dir

        # Args file.
        self.args_file = os.path.join(self.run_dir, 'args.json')

        # Model path managers.
        for models_name in self.models_names:
            mpm = ModelPathManager(self.run_dir, models_name)
            setattr(self.model_path_managers, models_name, mpm)

    def create_dirs(self):
        # Constant directories.
        create_dirs(CONST_DATA_DIR, CONST_MODELS_DIR, CONST_RUNS_DIR, CONST_TOKENIZERS_DIR)

        # Run dir.
        create_dir(self.run_dir)

        # Model path managers.
        for models_name in self.models_names:
            mpm = getattr(self, models_name)
            mpm.create_dirs()


class ModelPathManager():
    def __init__(self, run_dir, model_name):
        self.run_dir = run_dir
        self.model_name = model_name

        # Directories.
        self.model_dir = os.path.join(self.run_dir, model_name)
        self.checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        self.metrics_dir = os.path.join(run_dir, 'metrics')

        # Files.
        self.untrained_model_file = os.path.join(run_dir, 'untrained_model.pt')
        self.final_model_file = os.path.join(run_dir, 'final_model.pt')

    def create_dirs(self):
        create_dir(self.model_dir)
        create_dir(self.checkpoint_dir)
        create_dir(self.metrics_dir)
        create_dir(self.untrained_model_file)
        create_dir(self.final_model_file)
