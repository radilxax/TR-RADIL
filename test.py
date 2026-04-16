# test.py
"""Batch evaluation on external test set with optional TTA."""
import argparse
import os

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score, f1_score
)
import matplotlib.pyplot as plt

from dataset import CardiacMultiModalDataset
from model_mobile import Echo_RADIL
from config import Config
from torch.utils.data import DataLoader


def preprocess_test_csv(csv_path, data_dir, output_csv_path):
    """Add full_path column and filter missing files."""
    df = pd.read_csv(csv_path)
    df['full_path'] = df['filename'].apply(
        lambda x: os.path.join(data_dir, x + '.nii.gz')
    )
    exists = df['full_path'].apply(os.path.exists)
    print(f"  Files found: {exists.sum()}/{len(df)}")
    df = df[exists].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No valid samples found!")
    df.to_csv(output_csv_path, index=False)
    print(f"  Saved to {output_csv_path} ({len(df)} samples)")
    return output_csv_path


def run_inference(model, test_loader, device, tta_rounds=1):
    """Run inference with optional test-time augmentation."""
    all_probs_accumulated = None
    all_labels = None
    all_aux_preds = None
    all_aux_labels = None

    for tta_idx in range(tta_rounds):
        round_probs, round_labels = [], []
        round_aux_preds, round_aux_labels = [], []

        if tta_rounds > 1:
            print(f"  TTA Round {tta_idx + 1}/{tta_rounds}...")

        with torch.no_grad():
            for batch in test_loader:
                video, clinical, label, aux_label = batch
                video, clinical = video.to(device), clinical.to(device)

                # TTA transformations
                if tta_idx == 0:
                    v_input = video
                elif tta_idx == 1:
                    v_input = torch.flip(video, dims=[-1])   # horizontal flip
                elif tta_idx == 2:
                    v_input = torch.flip(video, dims=[-2])   # vertical flip
                elif tta_idx == 3:
                    v_input = video * 1.05                   # brightness +5%
                else:
                    v_input = video * 0.95                   # brightness -5%

                logits, aux_logits, _, _ = model(v_input, clinical)
                probs = F.softmax(logits, dim=1)
                round_probs.append(probs.cpu().numpy())
                round_labels.append(label.numpy())
                round_aux_preds.append(aux_logits.argmax(dim=1).cpu().numpy())
                round_aux_labels.append(aux_label.numpy())

        round_probs = np.concatenate(round_probs)
        round_labels = np.concatenate(round_labels)

        if all_probs_accumulated is None:
            all_probs_accumulated = round_probs
            all_labels = round_labels
            all_aux_preds = np.concatenate(round_aux_preds)
            all_aux_labels = np.concatenate(round_aux_labels)
        else:
            all_probs_accumulated += round_probs

    all_probs = all_probs_accumulated / tta_rounds
    all_preds = all_probs.argmax(axis=1)
    return all_preds, all_probs, all_labels, all_aux_preds, all_aux_labels


def main():
    parser = argparse.ArgumentParser(description="TR-RADIL External Test Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing NIfTI files")
    parser.add_argument("--save_dir", type=str, default="./test_results", help="Output directory")
    parser.add_argument("--no_tta", action="store_true", help="Disable TTA")
    parser.add_argument("--tta_rounds", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Preprocess test CSV
    processed_csv = os.path.join(args.save_dir, "test_processed.csv")
    if not os.path.exists(processed_csv):
        preprocess_test_csv(args.test_csv, args.data_dir, processed_csv)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = Echo_RADIL.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    model.eval()

    # Load dataset
    test_ds = CardiacMultiModalDataset(processed_csv, mode='test')
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )

    # Inference
    tta_rounds = 1 if args.no_tta else args.tta_rounds
    print(f"Running inference (TTA={'OFF' if args.no_tta else f'ON x{tta_rounds}'})...")
    all_preds, all_probs, all_labels, all_aux_preds, all_aux_labels = run_inference(
        model, test_loader, device, tta_rounds
    )

    # Results
    print("\n" + "=" * 60)
    print("  EXTERNAL TEST SET RESULTS")
    print("=" * 60)

    target_names = list(Config.LABEL_MAPPING.keys())
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:\n{cm}\n")

    auc = roc_auc_score(all_labels, all_probs[:, 1])
    ap = average_precision_score(all_labels, all_probs[:, 1])
    print(f"AUC-ROC: {auc:.4f}")
    print(f"AP: {ap:.4f}")

    # Optimal threshold search
    thresholds = np.arange(0.3, 0.7, 0.01)
    f1_scores = [f1_score(all_labels, (all_probs[:, 1] >= th).astype(int), average='weighted')
                 for th in thresholds]
    best_th = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)
    default_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"\nBest threshold: {best_th:.2f} (F1={best_f1:.4f} vs default F1={default_f1:.4f})")

    # Save results
    with open(os.path.join(args.save_dir, "test_results.txt"), 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"TTA: {'OFF' if args.no_tta else f'ON x{tta_rounds}'}\n")
        f.write(f"Samples: {len(all_labels)}\n\n")
        f.write(report + "\n")
        f.write(f"CM:\n{cm}\n\n")
        f.write(f"AUC: {auc:.4f}\nAP: {ap:.4f}\n")
        f.write(f"Best threshold: {best_th:.2f} (F1={best_f1:.4f})\n")

    # Plot curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    axes[0].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].set_title('ROC Curve'); axes[0].legend(); axes[0].grid(alpha=0.3)

    prec_arr, rec_arr, _ = precision_recall_curve(all_labels, all_probs[:, 1])
    axes[1].plot(rec_arr, prec_arr, linewidth=2, label=f'AP = {ap:.4f}')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('PR Curve'); axes[1].legend(); axes[1].grid(alpha=0.3)

    im = axes[2].imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=axes[2])
    axes[2].set_xticks(range(2)); axes[2].set_yticks(range(2))
    axes[2].set_xticklabels(target_names); axes[2].set_yticklabels(target_names)
    axes[2].set_xlabel('Predicted'); axes[2].set_ylabel('True')
    axes[2].set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(cm[i, j]), ha='center', va='center',
                         color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "test_curves.png"), dpi=150)

    # Save per-sample predictions
    results_df = test_ds.df.copy()
    results_df['pred_label'] = all_preds
    results_df['prob_worsen'] = all_probs[:, 1]
    results_df['correct'] = (all_preds == all_labels).astype(int)
    results_df.to_csv(os.path.join(args.save_dir, "test_predictions.csv"), index=False)

    print(f"\nAll results saved to {args.save_dir}")


if __name__ == '__main__':
    main()
