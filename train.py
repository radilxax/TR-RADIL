# train.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from dataset import CardiacDataModule
from model_mobile import Echo_RADIL
from config import Config
from pathlib import Path

def main():
    pl.seed_everything(42)
    
    dm = CardiacDataModule()
    dm.setup()
    
    model = Echo_RADIL()
    
    save_dir = Path(Config.SAVE_DIR)
    csv_logger = CSVLogger(save_dir=save_dir / 'logs', name="ECHO-RADIL")

    checkpoint_cb = ModelCheckpoint(
        dirpath=save_dir / 'checkpoints' / f"{csv_logger.name}_v{csv_logger.version}",
        filename='{epoch:02d}-f1_{val_f1_weighted:.4f}',
        monitor='val_f1_weighted',
        mode='max',
        save_top_k=1,
        save_last=True
    )
    
    early_stop = EarlyStopping(
        monitor='val_f1_weighted',
        patience=15,
        mode='max',
        min_delta=0.005
    )

    trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        accelerator='gpu',
        devices=[0], 
        callbacks=[checkpoint_cb, early_stop, LearningRateMonitor(logging_interval='epoch')],
        logger=csv_logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        benchmark=True
    )
    trainer.fit(model, dm)

    print(f"Best Model: {checkpoint_cb.best_model_path}")

if __name__ == '__main__':
    main()