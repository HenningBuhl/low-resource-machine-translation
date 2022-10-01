from distutils.util import strtobool

import os


# TODO argument names, default values, help strings
# TODO arguments that are not used in argparse but shared in notebooks (e.g. save_top_k or )

class ArgManager():
    '''A class that adds commonly shared arguments to the parser.'''

    def __init__(self):
        pass
    
    # Run.
    def add_run_args(self, parser):
        parser.add_argument('--dev-run', default=False, type=strtobool, help='Executes a fast dev run instead of fully training.')
        parser.add_argument('--fresh-run', default=False, type=strtobool, help='Ignores all cashed data on disk, reruns generation and overwrites everything.')
        parser.add_argument('--seed', default=0, type=int, help='The random seed of the program.')
        parser.add_argument('--eval-before-train', default=False, type=strtobool, help='Evaluate the model on the validation data before training.')

    # Metrics.
    def add_metrics_args(self, parser):
        parser.add_argument('--track-bleu', default=True, type=strtobool, help='Whether to track the SacreBLEU score metric.')
        parser.add_argument('--track-ter', default=False, type=strtobool, help='Whether to track the translation edit rate metric.')
        parser.add_argument('--track-tp', default=False, type=strtobool, help='Whether to track the translation perplexity metric.')
        parser.add_argument('--track-chrf', default=False, type=strtobool, help='Whether to track the CHRF score metric.')

    # Data.
    def add_data_args(self, parser):
        parser.add_argument('--shuffle-before-split', default=False, type=strtobool, help='Whether to shuffle the data before creating the train, validation and test sets.')
        parser.add_argument('--num-val-examples', default=3000, type=int, help='The number of validation examples.') 
        parser.add_argument('--num-test-examples', default=3000, type=int, help='The number of test examples.')

    # Tokenization.
    def add_tokenization_args(self, parser):
        parser.add_argument('--src-vocab-size', default=16000, type=int, help='The vocabulary size of the source language tokenizer.')
        parser.add_argument('--src-char-coverage', default=1.0, type=float, help='The character coverage (percentage) of the source language tokenizer.')
        parser.add_argument('--tgt-vocab-size', default=16000, type=int, help='The vocabulary size of the target language tokenizer.')
        parser.add_argument('--tgt-char-coverage', default=1.0, type=float, help='The character coverage (percentage) of the target language tokenizer.')

    # Architecture.
    def add_architecture_args(self, parser):
        parser.add_argument('--num-layers', default=6, type=int, help='The number of encoder and decoder layers.')
        parser.add_argument('--d-model', default=512, type=int, help='The embedding size.')
        parser.add_argument('--dropout', default=0.1, type=float, help='The dropout rate.')
        parser.add_argument('--num-heads', default=8, type=int, help='The number of attention heads.')
        parser.add_argument('--d-ff', default=2048, type=int, help='The feed forward dimension.')
        parser.add_argument('--max-len', default=128, type=int, help='The maximum sequence length.')

    # Optimizer.
    def add_optimizer_args(self, parser):
        parser.add_argument('--learning-rate', default=1e-4, type=float, help='The learning rate.')
        parser.add_argument('--weight-decay', default=0, type=float, help='The weight decay.')
        parser.add_argument('--beta-1', default=0.9, type=float, help='Beta_1 parameter of Adam.')
        parser.add_argument('--beta-2', default=0.98, type=float, help='Beta_2 parameter of Adam.')

    # Scheduler.
    def add_scheduler_args(self, parser):
        parser.add_argument('--enable-scheduling', default=False, type=strtobool, help='Whether to enable scheduling.')
        parser.add_argument('--warm-up-steps', default=4000, type=int, help='The number of warm up steps.')

    # Training.
    def add_training_args(self, parser):
        parser.add_argument('--batch-size', default=80, type=int, help='The batch size.')
        parser.add_argument('--label-smoothing', default=0, type=float, help='The amount of smoothing when calculating the loss.')
        parser.add_argument('--max-epochs', default=10, type=int, help='The maximum number of training epochs.')
        parser.add_argument('--max-examples', default=-1, type=int, help='The maximum number of training examples.')
        parser.add_argument('--shuffle-train-data', default=True, type=strtobool, help='Whether to shuffle the training data during training.')
        parser.add_argument('--gpus', default=1, type=int, help='The number of GPUs.')
        parser.add_argument('--num-workers', default=4, type=int, help='The number of pytorch workers.')
        parser.add_argument('--ckpt-path', default=None, type=str, help='The model checkpoint form which to resume training.')

    # Early Stopping + Model Checkpoint.
    def add_early_stopping_and_checkpoiting_args(self, parser):
        parser.add_argument('--enable-early-stopping', default=False, type=strtobool, help='Whether to enable early stopping.')
        parser.add_argument('--enable-checkpointing', default=False, type=strtobool, help='Whether to enable checkpointing. The best and the last version of the model are saved.')
        parser.add_argument('--monitor', default='val_loss', type=str, help='The metric to monitor.')
        parser.add_argument('--min-delta', default=0, type=float, help='The minimum change the metric must achieve.')
        parser.add_argument('--patience', default=3, type=int, help='Number of epochs that the monitored metric has time to improve.')
        parser.add_argument('--mode', default='min', type=str, choices=['min', 'max'], help='How the monitored metric should improve.')

    # TODO
    def auto_infer_args(self, args):
        '''Automatically infers arguments and sets them in the passed object.'''
        pass

    # TODO
    def sanity_check_args(self, args):
        '''Checks the validity of the arguments passed.'''
        pass

#parser.add_argument('--max-examples-step-2', default=-1, type=int, help='The maximum number of training examples in step 2.')  # NOTE: for step 2
#parser.add_argument('--ckpt-path-step-2', default=None, type=str, help='The model checkpoint form which to resume training of step 2.')  # NOTE: for step 2
