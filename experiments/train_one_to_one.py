# Add src module directory to system path for subsecuent imports.
import sys
sys.path.insert(0, '../src')

# From packages.
import os
import pytorch_lightning as pl
import argparse
from distutils.util import strtobool

# From repository.
from arg_manager import ArgManager
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
    arg_manager = ArgManager()
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
    arg_manager.add_run_args(parser)
    # Metrics.
    arg_manager.add_metrics_args(parser)
    # Data.
    arg_manager.add_data_args(parser)
    # Tokenization.
    arg_manager.add_tokenization_args(parser)
    # Architecture.
    arg_manager.add_architecture_args(parser)
    # Optimizer.
    arg_manager.add_optimizer_args(parser)
    # Scheduler.
    arg_manager.add_scheduler_args(parser)
    # Training.
    arg_manager.add_training_args(parser)
    # Early Stopping + Model Checkpoint.
    arg_manager.add_early_stopping_and_checkpointing_args(parser)

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
        model.receive_encoder(encoder_model)
        if not args.freeze_encoder:
            model.set_dropout(args.dropout, skip_decoder=True)
    if args.freeze_encoder:
        model.freeze_encoder()

    # Stitch decoder.
    if args.decoder_model_path is not None:
        decoder_model = load_model_from_path(args.decoder_model_path)
        model.receive_decoder(decoder_model)
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
        model_checkpoint = pl.callbacks.ModelCheckpoint(
            monitor=args.monitor,
            dirpath=checkpoint_dir,
            filename='{epoch}-{step}-{val_loss:.2f}',
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
            verbose=True,
        )
        callbacks.append(model_checkpoint)

    if args.enable_early_stopping:
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor=args.monitor,
            min_delta=args.min_delta,
            patience=args.patience,
            mode=args.mode,
            verbose=True,
        )
        callbacks.append(early_stopping_callback)

    if args.enable_scheduling:
        lr_monitor = pl.callbacks.LearningRateMonitor(
            logging_interval='step',
            log_momentum=True
        )
        callbacks.append(lr_monitor)

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
        model.load_from_checkpoint(model_checkpoint.best_model_path)

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
