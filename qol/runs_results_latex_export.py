import os
import sys
sys.path.insert(0, '../src')
import re

from util import create_dir, read_dict


def get_run_type(run_name):
    run_types = ['baseline', 'benchmark', 'cascaded', 'direct-pivoting', 'reverse-step-wise-pivoting', 'step-wise-pivoting'] # Order matters (from specific to general).
    for run_type in run_types:
        if run_type in run_name:
            return run_type
    raise ValueError(f'Directory {run_name} does not originate from known run type.')


def get_run_name(run_dir):
    m = re.search(r"\d", run_dir)
    return run_dir[0:m.start()-1]


def export_metrics(export_results_dir, metrics, model_name):
    metric_keys = [
        'test_loss_epoch',
        'train_loss_epoch',
        'train_loss_step',
        'val_loss_epoch',
        'test_score_epoch',
        'train_score_epoch',
        'train_score_step',
        'val_score_epoch',
        'greedy',
        'beam',
        'top_k',
        'top_p',
    ]

    for metric_key in metric_keys:
        if metric_key in metrics.keys():
            value_str = '\n'.join(map(str, metrics[metric_key]))
            if value_str != '':
                with open(os.path.join(export_results_dir, f'{model_name}.{metric_key}.txt'), 'w') as f:
                    f.write(value_str)


if __name__ == '__main__':
    # Directories.
    runs_dir = '../experiments/runs'
    export_results_dir = './results'
    create_dir(export_results_dir)

    # Iterate over all runs.
    for run_dir in os.listdir(runs_dir):
        # Get information.
        run_name = get_run_name(run_dir)
        run_dir = os.path.join(runs_dir, run_dir)
        run_type = get_run_type(run_name)

        # Export data depending on run type.
        if run_type == 'baseline' or run_type == 'cascaded':
            # Get meta data.
            hparams = read_dict(os.path.join(run_dir, 'hparams.json'))

            # Export metrics.
            metrics = read_dict(os.path.join(run_dir, 'results', 'metrics.json'))
            num_examples = hparams['max_examples']
            model_suffix = f'-{num_examples}-examples' if num_examples != -1 else ''
            export_metrics(export_results_dir, metrics, run_name + model_suffix)

        elif run_type == 'benchmark':
            for benchmark in os.listdir(run_dir):
                for model_name in os.listdir(os.path.join(run_dir, benchmark)):
                    metrics = read_dict(os.path.join(run_dir, benchmark, model_name, 'metrics.json'))
                    export_metrics(export_results_dir, metrics, f'{benchmark}.{model_name}')

        elif run_type == 'direct-pivoting':
            # Get meta data.
            hparams = read_dict(os.path.join(run_dir, 'hparams.json'))

            # Export metrics.
            metrics = read_dict(os.path.join(run_dir, 'results', 'metrics.json'))
            num_examples = hparams['max_examples']
            model_suffix = f'-{num_examples}-examples' if num_examples != -1 else ''
            export_metrics(export_results_dir, metrics, run_name + model_suffix)

            # Export pre-training metrics.
            metrics = read_dict(os.path.join(run_dir, 'pre-training-eval-results', 'metrics.json'))
            export_metrics(export_results_dir, metrics, run_name + '-untrained')
        
        elif run_type == 'step-wise-pivoting' or run_type == 'reverse-step-wise-pivoting':
            # Get meta data.
            hparams = read_dict(os.path.join(run_dir, 'hparams.json'))

            # Export metrics.
            metrics = read_dict(os.path.join(run_dir, 'results', 'metrics.json'))
            num_examples = hparams['max_examples_fine_tune']
            model_suffix = f'-{num_examples}-examples' if num_examples != -1 else ''
            export_metrics(export_results_dir, metrics, run_name + model_suffix)

            # Export pre-training metrics.
            metrics = read_dict(os.path.join(run_dir, 'pre-training-eval-results', 'metrics.json'))
            export_metrics(export_results_dir, metrics, run_name + '-untrained')

            # Export step-2 metrics.
            if hparams['step_two_model'] == None:
                metrics = read_dict(os.path.join(run_dir, 'step-results', 'metrics.json'))
                export_metrics(export_results_dir, metrics, run_name + '-step-2')
