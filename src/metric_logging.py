from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_only
from util import save_dict

import os
import time


class MetricLogger(LightningLoggerBase):
    '''A class logging metrics during training and testing.'''

    def __init__(self, track_score):
        super().__init__()
        self.track_score = track_score
        self.reset()

    @property
    def name(self):
        return 'MetricLogger'

    @property
    def version(self):
        return '0.1'

    @rank_zero_only
    def log_hyperparams(self, params):
        pass
        
    @rank_zero_only
    def log_metrics(self, metrics, step):
        for k in metrics.keys():
            if k in self.keys:
                self.metrics[k].append(metrics[k])

    def manual_save(self, dir, file):
        '''Saves all metrics in a combined json file and as separate txt files for each metric.'''
        
        end = time.time()
        training_time = end - self.start
        self.metrics['training_time'] = training_time

        # Save metrics dict as json.
        save_dict(file, self.metrics)
        
        # Save list associated with each key to txt file.
        for k in self.keys:
            value_str = '\n'.join(map(str, self.metrics[k]))
            with open(os.path.join(dir, f'{k}.txt'), 'w') as f:
                f.write(value_str)

    def reset(self):
        self.hparams = None
        self.keys = [
            'train_loss_step',
            'train_loss_epoch',
            'val_loss_step',
            'val_loss_epoch',
            'test_loss_step',
            'test_loss_epoch',
        ]
        if self.track_score:
            self.keys.extend([
                'train_score_step',
                'train_score_epoch',
                'val_score_step',
                'val_score_epoch',
                'test_score_step',
                'test_score_epoch',
            ])
        self.metrics = {k:[] for k in self.keys}
        self.start = time.time()
