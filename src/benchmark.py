from data import *
from path_management import CONST_MODELS_DIR, CONST_BENCHMARKS_DIR, get_dirs, get_files

import os
import sys
import gzip



class BenchmarkPreProcessor():
    def __init__(self):
        # Prepare benchmark configs.
        self.benchmarks = []
        for benchmark_name in get_dirs(CONST_BENCHMARKS_DIR):
            benchmark_folder = os.path.join(CONST_BENCHMARKS_DIR, benchmark_name)


        # Prepare model settings.
        self.models = []
        for model_name in get_dirs(CONST_MODELS_DIR):
            model_folder = os.join(CONST_MODELS_DIR, model_name)
            model_file = os.path.join(model_folder, 'model.pt')
            model_args_file = 
            self.models.append(ModelConfig())


class ModelConfig():
    def __init__(self,
                 type, # single, cascaded
                 langs, # [src, tgt] or [src, pvt, tgt]
                 paths, # model_path or paths
                 name=None,
                ):
        self.type = type
        self.langs = langs
        self.paths = paths
        if name is None:
            self.name = paths.split('/')[-1].replace('.pt', '')
        else:
            self.name = name


class BenchmarkConfig():
    def __init__(self, name):
        self.name = name
