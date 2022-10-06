import pytorch_lightning as pl


def get_model_checkpoint_callback(monitor, checkpoint_dir):
    return pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        dirpath=checkpoint_dir,
        filename='{epoch}-{step}-{val_loss:.2f}',
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        verbose=True,
    )

def get_early_stopping_callback(monitor, min_delta, patience, mode):
    return pl.callbacks.EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        mode=mode,
        verbose=True,
    )
  
def get_lr_monitor_callback():  # TODO test if this outputs anything to the console.
    return pl.callbacks.LearningRateMonitor(
        logging_interval='step',
        log_momentum=True
    )
