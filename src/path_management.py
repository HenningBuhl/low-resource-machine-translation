from util import *


CONST_BENCHMARKS_DIR = './benchmarks'
CONST_DATA_DIR = './data'
CONST_MODELS_DIR = './models'
CONST_RUNS_DIR = './runs'
CONST_TOKENIZERS_DIR = './tokenizers'


class ExperimentManager():
    '''A class managing an experiment. One experiment consists of training and managing one or more models.'''

    def __init__(self, run_name, *model_names=['model']):
        self.run_name = run_name

        # Run dir.
        self.run_dir = os.path.join(CONST_RUNS_DIR, f'{self.run_name}-{get_time_as_string()}')

        # Args file.
        self.args_file = os.path.join(self.run_dir, 'args.json')

        # Create model managers.
        self.model_managers = []
        for model_name in self.model_names:
            model_manager = ModelManager(self.run_dir, model_name)
            self.model_managers.append(model_manager)
            setattr(self, model_name, model_manager)

    def init(self):
        # Create constant directories.
        create_dirs(CONST_RUNS_DIR, CONST_TOKENIZERS_DIR)

        # Create run dir.
        create_dir(self.run_dir)

        # Initialize model managers.
        for model_manager in self.model_managers:
            model_manager.init()

    class ModelManager():
        def __init__(self, run_dir, model_name):
            self.model_name = model_name
            self.model_dir = os.path.join(self.run_dir, self.model_name)

            # Directories.
            self.checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')
            self.metrics_dir = os.path.join(self.model_dir, 'metrics')

            # Files.
            self.untrained_model_file = os.path.join(self.model_dir, 'model-untrained.pt')
            self.model_file = os.path.join(self.model_dir, 'model.pt')
            self.metrics_file = os.path.join(self.metrics_dir, 'metrics.json')
            self.metric_svg_template = os.path.join(self.metrics_dir, '{}.svg')

        def init(self):
            create_dir(self.model_dir)
            create_dir(self.checkpoint_dir)
            create_dir(self.metrics_dir)


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
