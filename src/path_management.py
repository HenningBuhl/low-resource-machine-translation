from util import *


CONST_DATA_DIR = './data'
CONST_MODELS_DIR = './models'
CONST_RUNS_DIR = './runs'
CONST_TOKENIZERS_DIR = './tokenizers'


class ExperimentPathManager():
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
            setattr(self, models_name, mpm)

    def init(self):
        # Constant directories.
        create_dirs(CONST_DATA_DIR, CONST_MODELS_DIR, CONST_RUNS_DIR, CONST_TOKENIZERS_DIR)

        # Run dir.
        create_dir(self.run_dir)

        # Model path managers.
        for models_name in self.models_names:
            mpm = getattr(self, models_name)
            mpm.init()


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

    def init(self):
        create_dir(self.model_dir)
        create_dir(self.checkpoint_dir)
        create_dir(self.metrics_dir)


class DataPathManager():
    def __init__(self, src_lang, tgt_lang, data_dir):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.data_dir = data_dir

        self.src_lang_dir = os.path.join(data_dir, src_lang)
        self.tgt_lang_dir = os.path.join(data_dir, tgt_lang)

        self.src_train_file = os.path.join(self.src_lang_dir, 'train.txt')
        self.tgt_train_file = os.path.join(self.tgt_lang_dir, 'train.txt')
        self.src_val_file = os.path.join(self.src_lang_dir, 'val.txt')
        self.tgt_val_file = os.path.join(self.tgt_lang_dir, 'val.txt')
        self.src_test_file = os.path.join(self.src_lang_dir, 'test.txt')
        self.tgt_test_file = os.path.join(self.tgt_lang_dir, 'test.txt')

        self.src_train_tokenized_file = os.path.join(self.src_lang_dir, 'train-tokenized.txt')
        self.tgt_train_tokenized_file = os.path.join(self.tgt_lang_dir, 'train-tokenized.txt')
        self.src_val_tokenized_file = os.path.join(self.src_lang_dir, 'val-tokenized.txt')
        self.tgt_val_tokenized_file = os.path.join(self.tgt_lang_dir, 'val-tokenized.txt')
        self.src_test_tokenized_file = os.path.join(self.src_lang_dir, 'test-tokenized.txt')
        self.tgt_test_tokenized_file = os.path.join(self.tgt_lang_dir, 'test-tokenized.txt')

        self.src_files = self.get_raw_files(self.src_lang_dir)
        self.tgt_files = self.get_raw_files(self.tgt_lang_dir)

    def get_raw_files(self, dir):
        files = get_files(dir)
        for f in ['train.txt', 'val.txt', 'test.txt', 'train-tokenized.txt', 'val-tokenized.txt', 'test-tokenized.txt']:
            if f in files:
              files.remove(f)
        return sorted([os.path.join(dir, f) for f in files])

class TokenizerPathManager():
    def __init__(self, lang, data_dir, mono_data_dir):
        self.tokenizer_path = os.path.join(CONST_TOKENIZERS_DIR, lang)
        self.tokenizer_sp_path = os.path.join(self.tokenizer_path, lang)

        # Files for training.
        train_file = os.path.join(data_dir, lang , 'train.txt')
        val_file = os.path.join(data_dir, lang , 'val.txt')
        mono_data_files = []
        if mono_data_dir is not None:
            mono_data_files = get_files(mono_data_dir)
        self.files = [train_file] + [val_file] + mono_data_files

def get_files(dir):
    return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
