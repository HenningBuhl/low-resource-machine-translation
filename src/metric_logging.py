from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_only
from util import save_dict

import os
import time


class MetricLogger(LightningLoggerBase):
    '''A class logging metrics during training and testing.'''

    def __init__(self):
        super().__init__()
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
            if k == 'epoch':  # Skip epoch key.  TODO is it still there? see TODO below...
                continue
            if k in self.metrics.keys():
                self.metrics[k].append(metrics[k])
            else:
                self.metrics[k] = [metrics[k]]

    def manual_save(self, dir):
        '''Saves all metrics in a combined json file and as separate txt files for each metric.'''
        
        # Remove epoch key (TODO why is it there in the first place? some later it just disappeared...).
        #del self.metrics['epoch']
        
        # Save list associated with each key to txt file.
        for k in self.metrics.keys():
            value_str = '\n'.join(map(str, self.metrics[k]))
            with open(os.path.join(dir, f'{k}.txt'), 'w') as f:
                f.write(value_str)
        
        # Measuring training time.
        end = time.time()
        elapsed_time = end - self.start
        self.metrics['elapsed_time'] = elapsed_time

        # Save metrics dict as json.
        save_dict(os.path.join(dir, 'metrics.json'), self.metrics)

    def reset(self):
        self.metrics = {}
        self.start = time.time()
