# Add src module directory to system path for subsecuent imports.
import sys
sys.path.insert(0, '../src')

# From packages.
import os
import pytorch_lightning as pl
import argparse
from distutils.util import strtobool

# From repository.
from callbacks import *
from constants import *
from data import ParallelDataPreProcessor
from metric_logging import MetricLogger
from plotting import plot_metric
from tokenizer import TokenizerBuilder
from transformer import *
from util import *


def main():
    ########################
    # Arguments.
    ########################

    # Define arguments with argparse.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Experiment.
    parser.add_argument('--name', default='baseline', type=str, help='The name of the experiment.')
    parser.add_argument('--src-lang', default='de', type=str, help='The source language.')
    parser.add_argument('--tgt-lang', default='en', type=str, help='The target language.')
    parser.add_argument('--encoder-model-path', default=None, type=str, help='The folder of the model form which to use the encoder.')
    parser.add_argument('--decoder-model-path', default=None, type=str, help='The folder of the model from which to use the decoder.')
    parser.add_argument('--freeze-encoder', default=False, type=strtobool, help='Whether to freeze the encoder.')
    parser.add_argument('--freeze-decoder', default=False, type=strtobool, help='Whether to freeze the decoder.')

    # Run.
    parser.add_argument('--dev-run', default=False, type=strtobool, help='Executes a fast dev run instead of fully training.')
    parser.add_argument('--fresh-run', default=False, type=strtobool, help='Ignores all cashed data on disk, reruns generation and overwrites everything.')
    parser.add_argument('--seed', default=0, type=int, help='The random seed of the program.')

    # Metrics.
    parser.add_argument('--track-bleu', default=True, type=strtobool, help='Whether to track the SacreBLEU score metric.')
    parser.add_argument('--track-ter', default=False, type=strtobool, help='Whether to track the translation edit rate metric.')
    parser.add_argument('--track-chrf', default=False, type=strtobool, help='Whether to track the CHRF score metric.')

    # Data.
    parser.add_argument('--shuffle-before-split', default=False, type=strtobool, help='Whether to shuffle the data before creating the train, validation and test sets.')
    parser.add_argument('--num-val-examples', default=3000, type=int, help='The number of validation examples.') 
    parser.add_argument('--num-test-examples', default=3000, type=int, help='The number of test examples.')

    # Tokenization.
    parser.add_argument('--src-vocab-size', default=16000, type=int, help='The vocabulary size of the source language tokenizer.')
    parser.add_argument('--src-char-coverage', default=1.0, type=float, help='The character coverage (percentage) of the source language tokenizer.')
    parser.add_argument('--tgt-vocab-size', default=16000, type=int, help='The vocabulary size of the target language tokenizer.')
    parser.add_argument('--tgt-char-coverage', default=1.0, type=float, help='The character coverage (percentage) of the target language tokenizer.')

    # Architecture.
    parser.add_argument('--num-layers', default=6, type=int, help='The number of encoder and decoder layers.')
    parser.add_argument('--d-model', default=512, type=int, help='The embedding size.')
    parser.add_argument('--dropout', default=0.1, type=float, help='The dropout rate.')
    parser.add_argument('--num-heads', default=8, type=int, help='The number of attention heads.')
    parser.add_argument('--d-ff', default=2048, type=int, help='The feed forward dimension.')
    parser.add_argument('--max-len', default=128, type=int, help='The maximum sequence length.')

    # Optimizer.
    parser.add_argument('--learning-rate', default=1e-4, type=float, help='The learning rate.')
    parser.add_argument('--weight-decay', default=0, type=float, help='The weight decay.')
    parser.add_argument('--beta-1', default=0.9, type=float, help='Beta_1 parameter of Adam.')
    parser.add_argument('--beta-2', default=0.98, type=float, help='Beta_2 parameter of Adam.')

    # Scheduler.
    parser.add_argument('--enable-scheduling', default=False, type=strtobool, help='Whether to enable scheduling.')
    parser.add_argument('--warm-up-steps', default=4000, type=int, help='The number of warm up steps.')

    # Training.
    parser.add_argument('--batch-size', default=80, type=int, help='The batch size.')
    parser.add_argument('--label-smoothing', default=0.0, type=float, help='The amount of smoothing when calculating the loss.')
    parser.add_argument('--max-epochs', default=10, type=int, help='The maximum number of training epochs.')
    parser.add_argument('--max-examples', default=-1, type=int, help='The maximum number of training examples.')
    parser.add_argument('--shuffle-train-data', default=False, type=strtobool, help='Whether to shuffle the training data during training.')
    parser.add_argument('--gpus', default=1, type=int, help='The number of GPUs used.')
    parser.add_argument('--num-workers', default=4, type=int, help='The number of pytorch workers.')
    parser.add_argument('--ckpt-path', default=None, type=str, help='The model checkpoint form which to resume training.')
    parser.add_argument('--eval-before-train', default=False, type=strtobool, help='Evaluate the model on the validation data before training.')

    # Early Stopping + Model Checkpoint.
    parser.add_argument('--enable-early-stopping', default=False, type=strtobool, help='Whether to enable early stopping.')
    parser.add_argument('--enable-checkpointing', default=False, type=strtobool, help='Whether to enable checkpointing. The best and the last version of the model are saved.')
    parser.add_argument('--monitor', default='val_loss', type=str, help='The metric to monitor.')
    parser.add_argument('--min-delta', default=0, type=float, help='The minimum change the metric must achieve.')
    parser.add_argument('--patience', default=3, type=int, help='Number of epochs that the monitored metric has time to improve.')
    parser.add_argument('--mode', default='min', type=str, choices=['min', 'max'], help='How the monitored metric should improve.')

    # Parse args.
    args = parser.parse_args()

    # Add model type to args.
    args.model_type = 'one-to-one'

    # Print args.
    print(f'Arguments:')
    print(args)

    ########################
    # Setup.
    ########################

    # Set seed.
    set_seed(args.seed)

    # Create runs and tokenizers dirs.
    create_dirs(CONST_RUNS_DIR, CONST_TOKENIZERS_DIR)

    # Create run dir.
    run_dir = os.path.join(CONST_RUNS_DIR, f'{args.name}-{get_time_as_string()}')
    create_dir(run_dir)

    # Create model dirs.
    model_dir = os.path.join(run_dir, args.name)
    create_dir(model_dir)
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    create_dir(checkpoint_dir)
    metrics_dir = os.path.join(model_dir, 'metrics')
    create_dir(metrics_dir)

    # Save arguments.
    save_dict(os.path.join(run_dir, 'args.json'), args.__dict__)

    ########################
    # Data.
    ########################

    # Create ParallelDataPreProcessor.
    pp = ParallelDataPreProcessor(args.src_lang, args.tgt_lang)

    # Split data into (train, val, test) sets.
    pp.split_data(args.shuffle_before_split, args.num_val_examples, args.num_test_examples, args.fresh_run)

    # Load tokenizers.
    src_tokenizer = TokenizerBuilder(args.src_lang, args.tgt_lang).build(
        args.src_vocab_size, args.src_char_coverage, fresh_run=args.fresh_run)
    tgt_tokenizer = TokenizerBuilder(args.tgt_lang, args.src_lang).build(
        args.tgt_vocab_size, args.tgt_char_coverage, fresh_run=args.fresh_run)

    # Load dataloaders.
    train_dataloader, val_dataloader, test_dataloader = pp.pre_process(
        src_tokenizer, tgt_tokenizer, args.batch_size, args.shuffle_train_data,
        args.max_examples, args.max_len, fresh_run=args.fresh_run)

    ########################
    # Model.
    ########################

    # Create model.
    model = Transformer(
        src_tokenizer,
        tgt_tokenizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        enable_scheduling=args.enable_scheduling,
        warm_up_steps=args.warm_up_steps,
        num_layers=args.num_layers,
        d_model=args.d_model,
        dropout=args.dropout,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_len=args.max_len,
        label_smoothing=args.label_smoothing,
        track_bleu=args.track_bleu,
        track_ter=args.track_ter,
        track_chrf=args.track_chrf,
    )

    # Stitch encoder.
    if args.encoder_model_path is not None:
        encoder_model = load_model_from_path(args.encoder_model_path)
        model.set_encoder(encoder_model)
        if not args.freeze_encoder:
            model.set_dropout(args.dropout, skip_decoder=True)
    if args.freeze_encoder:
        model.freeze_encoder()

    # Stitch decoder.
    if args.decoder_model_path is not None:
        decoder_model = load_model_from_path(args.decoder_model_path)
        model.set_decoder(decoder_model)
        if not args.freeze_decoder:
            model.set_dropout(args.dropout, skip_encoder=True)
    if args.freeze_decoder:
        model.freeze_decoder()

    # Save untrained model.
    model.save(os.path.join(model_dir, 'model-untrained.pt'))

    ########################
    # Training.
    ########################

    # Create callbacks and loggers.
    callbacks = []
    if args.enable_checkpointing:
        mcc = get_model_checkpoint_callback(args.monitor, checkpoint_dir)
        callbacks.append(mcc)

    if args.enable_early_stopping:
        callbacks.append(get_early_stopping_callback(args.monitor, args.min_delta, args.patience, args.mode))

    if args.enable_scheduling:
        callbacks.append(get_lr_monitor_callback())

    # Create metric logger.
    metric_logger = MetricLogger()

    # Create trainer.
    trainer = pl.Trainer(
        deterministic=True,
        fast_dev_run=args.dev_run,
        max_epochs=args.max_epochs,
        logger=metric_logger,
        log_every_n_steps=1,
        enable_checkpointing=args.enable_checkpointing,
        default_root_dir=checkpoint_dir,
        callbacks=callbacks,
        gpus=args.gpus if str(device) == 'cuda' else 0
    )

    # Evaluate before training.
    if args.eval_before_train:
        trainer.validate(model, dataloaders=val_dataloader)

    # Training.
    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=args.ckpt_path)

    # Save model.
    if args.enable_checkpointing:
        model.load_from_checkpoint(mcc.best_model_path)
    model.save(os.path.join(model_dir, 'model.pt'))

    # Testing.
    trainer.test(model, dataloaders=test_dataloader)

    # Save recorded metrics.
    metric_logger.manual_save(metrics_dir)

    # Save metric plots.
    for metric in model.tracked_metrics:
        plot_metric(metric_logger.metrics, metric,
                    save_path=os.path.join(metrics_dir, '{}.svg').format(metric))


if __name__ == '__main__':
    main()
