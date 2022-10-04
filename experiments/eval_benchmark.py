# Add src module directory to system path for subsecuent imports.
import sys
sys.path.insert(0, '../src')

# From packages.
import os
import pytorch_lightning as pl
import argparse
import torchmetrics
import torch
from distutils.util import strtobool

# From repository.
from arg_manager import *
from constants import *
from data import *
from layers import *
from metric_logging import *
from plotting import *
from tokenizer import *
from transformer import *
from util import *


def main():
    ########################
    # Arguments.
    ########################

    # Define arguments with argparse.
    arg_manager = ArgManager()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Experiment.
    parser.add_argument('--seed', default=0, type=int, help='The random seed of the program.')
    parser.add_argument('--use-greedy', default=True, type=strtobool, help='Whether to use the greedy inference method to evaluate.')
    parser.add_argument('--use-beam-search', default=False, type=strtobool, help='Whether to use the beam search inference method to evaluate.')
    parser.add_argument('--beam-size', default=[8], type=int, nargs="*", help='The number of different beam sizes to be used.')
    parser.add_argument('--use-top-k', default=False, type=strtobool, help='Whether to use the top-K inference method to evaluate.')
    parser.add_argument('--top-k', default=[15], type=int, nargs="*", help='The differnt top-Ks being used.')
    parser.add_argument('--use-top-p', default=False, type=strtobool, help='Whether to use the top-p (nucleus) inference method to evaluate.')
    parser.add_argument('--top-p', default=[0.7], type=float, nargs="*", help='The differnt top-ps being used.')

    # Metrics.
    arg_manager.add_metrics_args(parser)

    # Parse args.
    args = parser.parse_args()

    # Print args.
    print(f'Arguments:')
    print(args)

    ########################
    # Setup.
    ########################

    # Set seed.
    from pytorch_lightning import seed_everything
    seed_everything(args.seed, workers=True)

    # Create constant dirs.
    create_dirs(CONST_BENCHMARKS_DIR, CONST_DATA_DIR, CONST_MODELS_DIR, CONST_RUNS_DIR, CONST_TOKENIZERS_DIR)

    # Create run dir.
    run_dir = os.path.join(CONST_RUNS_DIR, f'benchmark-{get_time_as_string()}')
    create_dir(run_dir)

    # Create results dir.
    results_dir = os.path.join(run_dir, 'results')
    create_dir(results_dir)

    # Save arguments.
    save_dict(os.path.join(run_dir, 'args.json'), args.__dict__)

    # Which inference methods to perform.
    runs = [] # (method, kwargs)
    if args.use_greedy:
        runs.append(('greedy', {}))
    if args.use_beam_search:
        for beam_size in args.beam_size:
            runs.append(('beam_search', {'beam_size': beam_size}))
    if args.use_top_k:
        for top_k in args.top_k:
            runs.append(('sampling', {'top_k': top_k}))
    if args.use_top_p:
        for top_p in args.top_p:
            runs.append(('sampling', {'top_p': top_p}))
    print(f'The following inference configs are used: {runs}')

    # Which metrics to record.
    metrics = {}
    if args.track_bleu:
        metrics['bleu'] = torchmetrics.SacreBLEUScore(tokenize='char')
    if args.track_ter:
        metrics['ter'] = torchmetrics.TranslationEditRate()
    if args.track_chrf:
        metrics['chrf'] = torchmetrics.CHRFScore()
    print(f'The following metrics are recorded: {metrics}')

    ########################
    # Benchmark.
    ########################

    # Iterate over benchmarks.
    for benchmark_name in get_dirs(CONST_BENCHMARKS_DIR):
        print(f'Benchmark: {benchmark_name}')

        # Create benchmark data preprocessor.
        pp = BenchmarkDataPreProcessor(os.path.join(CONST_BENCHMARKS_DIR, benchmark_name))

        # Iterate over models.
        for model_name in get_dirs(CONST_MODELS_DIR):
            print(f'Model: {model_name}')

            # Load model args.
            args = load_dict(os.path.join(CONST_MODELS_DIR, model_name, 'args.json'))

            # Load source and target sentences.
            src_sentences, tgt_sentences = pp.get_src_tgt_sentences(args.src_lang, args.tgt_lang)
            reference_sentences = [[t] for t in tgt_sentences]

            # Load tokenizers and model(s).
            inference_fn = None
            model_type = args.model_type
            if model_type == 'cascaded':
                # Load tokenizers.
                src_tokenizer = TokenizerBuilder(args.src_lang).build()
                pvt_tokenizer = TokenizerBuilder(args.pvt_lang).build()
                tgt_tokenizer = TokenizerBuilder(args.tgt_lang).build()

                # Load models.
                src_pvt_model = load_model_from_path(args.src_pvt_model_path, src_tokenizer, pvt_tokenizer)
                pvt_tgt_model = load_model_from_path(args.pvt_tgt_model_path, pvt_tokenizer, tgt_tokenizer)

                # Create function that translates input text (model_type agnostic for further code below).
                inference_fn = lambda text, method, kwargs : cascaded_inference(text, src_pvt_model, pvt_tgt_model, method, **kwargs)
            elif model_type == 'one-to-one':
                # Load tokenizers.
                src_tokenizer = TokenizerBuilder(args.src_lang).build()
                tgt_tokenizer = TokenizerBuilder(args.tgt_lang).build()

                # Load model.
                model = load_model_from_path(os.path.join(CONST_MODELS_DIR, model_name), src_tokenizer, tgt_tokenizer)

                # Create function that translates input text (model_type agnostic for further code below).
                inference_fn = lambda text, method, kwargs : model.translate(text, method, **kwargs)
            else:
                raise Exception(f'Unknown model type {model_type}.')

            # Iterate over inference methods.
            for method, kwargs in runs:
                print(f'Method: {method} with args: {kwargs}')

                # Translate all source sentences.
                translations = []
                for text in src_sentences:
                    translation = inference_fn(text, method, kwargs)
                    translations.append(translation)

                # Save the translations produces with the medhot and kwargs.
                with open(os.path.join(results_dir, f'{benchmark_name}.{model_name}.{method}{arg_str}.translations.txt'), 'w', encoding='utf8') as f:
                    f.write('n'.join(translations))

                # Iterate over all metrics, calcualte them and save the result.
                for metric, metric_fn in metrics.items():
                    score = metric_fn(translations, reference_sentences).item()
                    arg_str = '' if len(kwargs) == 0 else ('-' + '-'.join([f'{key}={value}' for key, value in kwargs.items()]))
                    with open(os.path.join(results_dir, f'{benchmark_name}.{model_name}.{metric}.{method}{arg_str}.txt'), 'w') as f:
                        f.write(str(score))


if __name__ == '__main__':
    main()
