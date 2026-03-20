# test.py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score, f1_score
)
import matplotlib.pyplot as plt
import os

from dataset import CardiacMultiModalDataset
from model_mobile import Echo_RADIL
from config import Config
from torch.utils.data import DataLoader

CKPT_PATH = "/home/redili/TR_project/code_2_4/code_2class/save_ECHO-RADIL_no3/checkpoints/ECHO-RADIL_v1/epoch=31-f1_val_f1_weighted=0.7002.ckpt"
TEST_XLSX = "/home/redili/TR_project/code_2_4/data_no3/test_external.csv"
TEST_DATA_DIR = "/hdd/ScholarSupport/1-ZhongshanHeart/Basic/data_2_4_new/center_xiamen"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "/home/redili/TR_project/code_2_4/code_2class/test_results_echoradil_no3_v1"

ENABLE_TTA = True
TTA_ROUNDS = 5

def preprocess_test_xlsx(csv_path, data_dir, output_csv_path):
    df = pd.read_csv(csv_path)
    suffix = '.nii.gz'
    df['full_path'] = df['filename'].apply(lambda x: os.path.join(data_dir, x + suffix))
    
    exists = df['full_path'].apply(os.path.exists)
    print(f"  Files found: {exists.sum()}/{len(df)}")
    
    if not exists.all():
        missing_count = (~exists).sum()
        print(f"   {missing_count} files not found, keeping {exists.sum()} valid samples")
        df = df[exists].reset_index(drop=True)
    
    if len(df) == 0:
        raise RuntimeError("No valid samples found!")
    
    df.to_csv(output_csv_path, index=False)
    print(f"  ✅ Saved to {output_csv_path} ({len(df)} samples)")
    return output_csv_path

def run_inference(model, test_loader, device, tta_rounds=1):
    all_probs_accumulated = None
    all_labels = None
    all_aux_preds = None
    all_aux_labels = None
    
    for tta_idx in range(tta_rounds):
        round_probs = []
        round_labels = []
        round_aux_preds = []
        round_aux_labels = []
        
        if tta_rounds > 1:
            print(f"  TTA Round {tta_idx + 1}/{tta_rounds}...")
        
        with torch.no_grad():
            for batch in test_loader:
                video, clinical, label, aux_label = batch
                video = video.to(device)
                clinical = clinical.to(device)
                
                if tta_idx == 0:
                    v_input = video
                elif tta_idx == 1:
                    v_input = torch.flip(video, dims=[-1])
                elif tta_idx == 2:
                    v_input = torch.flip(video, dims=[-2])
                elif tta_idx == 3:
                    v_input = video * 1.05
                else:
                    v_input = video * 0.95
                
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
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    processed_csv = os.path.join(SAVE_DIR, "test_external_processed.csv")
    if not os.path.exists(processed_csv):
        print("Preprocessing test xlsx...")
        preprocess_test_xlsx(TEST_XLSX, TEST_DATA_DIR, processed_csv)
    else:
        print(f"Using existing: {processed_csv}")
    
    print(f"Loading model from {CKPT_PATH}...")
    model = Echo_RADIL.load_from_checkpoint(CKPT_PATH)
    model.to(DEVICE)
    model.eval()
    
    print("Loading test dataset...")
    test_ds = CardiacMultiModalDataset(processed_csv, mode='test')
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    tta_rounds = TTA_ROUNDS if ENABLE_TTA else 1
    print(f"Running inference (TTA={'ON x' + str(tta_rounds) if ENABLE_TTA else 'OFF'})...")
    
    all_preds, all_probs, all_labels, all_aux_preds, all_aux_labels = run_inference(
        model, test_loader, DEVICE, tta_rounds
    )
    
    print("\n" + "=" * 60)
    print(" TEST SET RESULTS")
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
    
    aux_acc = np.mean(all_aux_preds == all_aux_labels)
    print(f"Aux Accuracy: {aux_acc:.4f}")
    
    thresholds = np.arange(0.3, 0.7, 0.01)
    f1_scores = [f1_score(all_labels, (all_probs[:, 1] >= th).astype(int), average='weighted') for th in thresholds]
    best_th = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)
    default_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"\n🎯 Best threshold: {best_th:.2f} (F1={best_f1:.4f} vs default F1={default_f1:.4f})")
    
    optimal_preds = (all_probs[:, 1] >= best_th).astype(int)
    print(f"\n📊 Results with optimal threshold ({best_th:.2f}):")
    print(classification_report(all_labels, optimal_preds, target_names=target_names, digits=4))
    
    with open(os.path.join(SAVE_DIR, "test_results.txt"), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint: {CKPT_PATH}\n")
        f.write(f"TTA: {'ON x' + str(tta_rounds) if ENABLE_TTA else 'OFF'}\n")
        f.write(f"Samples: {len(all_labels)}\n")
        f.write("=" * 60 + "\n\n")
        f.write("=== Default threshold (0.5) ===\n")
        f.write(report + "\n")
        f.write(f"CM:\n{cm}\n\n")
        f.write(f"AUC: {auc:.4f}\nAP: {ap:.4f}\n\n")
        f.write(f"=== Optimal threshold ({best_th:.2f}) ===\n")
        f.write(classification_report(all_labels, optimal_preds, target_names=target_names, digits=4))
    
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
                        color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "test_curves.png"), dpi=150)
    
    results_df = test_ds.df.copy()
    results_df['pred_label'] = all_preds
    results_df['prob_worsen'] = all_probs[:, 1]
    results_df['correct'] = (all_preds == all_labels).astype(int)
    results_df.to_csv(os.path.join(SAVE_DIR, "test_predictions.csv"), index=False)
    
    print(f"\n All results saved to {SAVE_DIR}")
    print(" Testing complete!")

if __name__ == '__main__':
    main()