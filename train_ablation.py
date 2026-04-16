# train_ablation.py
"""Run ablation experiments by toggling model components."""
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from dataset import CardiacDataModule
from model_ablation import Echo_RADIL_Ablation
from config import Config
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="TR-RADIL Ablation Study")
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--no_spatial_attn', action='store_true', help='Remove spatial attention')
    parser.add_argument('--no_temporal_transformer', action='store_true', help='Remove temporal transformer')
    parser.add_argument('--no_temporal_attn', action='store_true', help='Remove temporal attention pooling')
    parser.add_argument('--no_aux_task', action='store_true', help='Remove auxiliary task')
    parser.add_argument('--no_clinical', action='store_true', help='Remove clinical features')
    parser.add_argument('--no_interaction', action='store_true', help='Remove interaction features (diff/prod)')
    parser.add_argument('--no_interval_embed', action='store_true', help='Remove interval bucket embedding')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    ablation_config = {
        'use_spatial_attn': not args.no_spatial_attn,
        'use_temporal_transformer': not args.no_temporal_transformer,
        'use_temporal_attn': not args.no_temporal_attn,
        'use_aux_task': not args.no_aux_task,
        'use_clinical': not args.no_clinical,
        'use_interaction': not args.no_interaction,
        'use_interval_embed': not args.no_interval_embed,
    }

    print(f"\n{'=' * 60}")
    print(f"Ablation Experiment: {args.exp_name}")
    print(f"{'=' * 60}")
    for k, v in ablation_config.items():
        print(f"  {'ON ' if v else 'OFF'} {k}")
    print(f"{'=' * 60}\n")

    dm = CardiacDataModule()
    dm.setup()

    model = Echo_RADIL_Ablation(**ablation_config)

    save_dir = Path(Config.SAVE_DIR)
    csv_logger = CSVLogger(save_dir=save_dir / 'logs_ablation', name=args.exp_name)

    checkpoint_cb = ModelCheckpoint(
        dirpath=save_dir / 'checkpoints_ablation' / args.exp_name,
        filename='{epoch:02d}-f1_{val_f1_weighted:.4f}',
        monitor='val_f1_weighted', mode='max', save_top_k=1, save_last=True
    )

    early_stop = EarlyStopping(
        monitor='val_f1_weighted', patience=15, mode='max', min_delta=0.005
    )

    trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        accelerator='gpu',
        devices=[args.gpu],
        callbacks=[checkpoint_cb, early_stop, LearningRateMonitor(logging_interval='epoch')],
        logger=csv_logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        benchmark=True
    )
    trainer.fit(model, dm)

    print(f"\nBest Model: {checkpoint_cb.best_model_path}")
    print(f"Best F1: {checkpoint_cb.best_model_score:.4f}")

    summary_path = save_dir / 'ablation_results.txt'
    with open(summary_path, 'a') as f:
        f.write(f"{args.exp_name}\t{checkpoint_cb.best_model_score:.4f}\t{ablation_config}\n")
    print(f"Summary appended to {summary_path}")


if __name__ == '__main__':
    main()
