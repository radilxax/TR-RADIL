# inference.py
"""
TR-RADIL Single-Patient Inference Script.

Usage:
    python inference.py \
        --video_path data/patient_001.nii.gz \
        --age 65 --sex female --baseline_tr_grade 2 \
        --prediction_horizon_days 365 \
        --checkpoint_path weights/best_model.ckpt
"""
import argparse
import os
import sys
from typing import Any, cast

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model_mobile import Echo_RADIL
from config import Config


def load_video(video_path: str) -> np.ndarray:
    """Load NIfTI video -> (T, H, W) float32."""
    nii = cast(Any, nib.load(video_path))
    header_dtype = nii.header.get_data_dtype()
    if header_dtype.names is not None and 'R' in header_dtype.names:
        raw = np.asanyarray(nii.dataobj)
        v = 0.299 * raw['R'].astype(np.float32) + \
            0.587 * raw['G'].astype(np.float32) + \
            0.114 * raw['B'].astype(np.float32)
    else:
        v = nii.get_fdata(dtype=np.float32)

    v = np.squeeze(v)
    if v.ndim == 4 and v.shape[-1] == 3:
        v = 0.299 * v[..., 0] + 0.587 * v[..., 1] + 0.114 * v[..., 2]
    if v.ndim == 2:
        v = v[np.newaxis, :, :]
    elif v.ndim == 3:
        v = np.transpose(v, (2, 0, 1))
    v = np.ascontiguousarray(v)

    T_raw = v.shape[0]
    if T_raw >= Config.NUM_FRAMES:
        start = (T_raw - Config.NUM_FRAMES) // 2
        v = v[start: start + Config.NUM_FRAMES]
    else:
        padding = np.tile(v[-1:], (Config.NUM_FRAMES - T_raw, 1, 1))
        v = np.concatenate([v, padding], axis=0)
    return v


def preprocess_frames(video_raw: np.ndarray, is_external: bool = False) -> torch.Tensor:
    """Per-frame preprocessing -> (3, T, H, W) tensor."""
    aug = A.Compose([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                     max_pixel_value=255.0),
        ToTensorV2(),
    ])
    global_min, global_max = video_raw.min(), video_raw.max()
    frames = []

    for i in range(video_raw.shape[0]):
        fr = video_raw[i]
        if is_external:
            fr = np.ascontiguousarray(np.flip(np.rot90(fr, k=1), axis=0))

        # Center crop 70%
        h, w = fr.shape[:2]
        hc, wc = h // 2, w // 2
        hh, ww = int(h * 0.7) // 2, int(w * 0.7) // 2
        if hh > 0 and ww > 0:
            fr = fr[hc - hh: hc + hh, wc - ww: wc + ww]

        fr = cv2.resize(fr, (Config.IMG_SIZE, Config.IMG_SIZE))

        # Min-max -> uint8
        if fr.dtype != np.uint8:
            if global_max > global_min:
                fr = (255.0 * (fr - global_min) / (global_max - global_min)).astype(np.uint8)
            else:
                fr = np.zeros_like(fr, dtype=np.uint8)

        fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2RGB)

        # Edge masking
        h2, w2 = fr.shape[:2]
        fr[:, :w2 // 10] = 0
        pad = int(0.03 * min(h2, w2))
        if pad > 0:
            fr[:pad, :] = 0; fr[-pad:, :] = 0; fr[:, -pad:] = 0

        frames.append(aug(image=fr)['image'])

    return torch.stack(frames).permute(1, 0, 2, 3)


def encode_clinical(age, sex, baseline_tr_grade, prediction_horizon_days):
    """Encode clinical variables -> (4,) tensor."""
    return torch.tensor([
        (baseline_tr_grade - 1.0) / (Config.NUM_SEVERITY_LEVELS - 1.0),
        age / 100.0,
        1.0 if sex.lower() in ('female', 'f') else 0.0,
        prediction_horizon_days / 3650.0,
    ], dtype=torch.float32)


def apply_tta(video, idx):
    if idx == 0: return video
    if idx == 1: return torch.flip(video, dims=[-1])
    if idx == 2: return torch.flip(video, dims=[-2])
    if idx == 3: return video * 1.05
    return video * 0.95


@torch.no_grad()
def predict(video_path, age, sex, baseline_tr_grade, prediction_horizon_days,
            checkpoint_path, is_external=False, enable_tta=True, tta_rounds=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Echo_RADIL.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device).eval()

    video_t = preprocess_frames(load_video(video_path), is_external).unsqueeze(0).to(device)
    clinical_t = encode_clinical(age, sex, baseline_tr_grade, prediction_horizon_days).unsqueeze(0).to(device)

    rounds = tta_rounds if enable_tta else 1
    acc_probs = None
    for i in range(rounds):
        logits, _, _, _ = model(apply_tta(video_t, i), clinical_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        acc_probs = probs if acc_probs is None else acc_probs + probs

    final_probs = acc_probs / rounds
    prob_worsen = float(final_probs[0, 1])

    # Auxiliary prediction (from clean pass)
    _, aux_logits, _, _ = model(video_t, clinical_t)
    aux_pred = int(torch.argmax(aux_logits, dim=1).item()) + 1

    grade_map = {1: "Mild", 2: "Moderate", 3: "Severe"}
    return {
        "progression_probability": round(prob_worsen, 4),
        "prediction": "Worsen" if prob_worsen >= 0.5 else "Stable/Improved",
        "predicted_followup_grade": aux_pred,
        "predicted_followup_severity": grade_map.get(aux_pred, str(aux_pred)),
        "prediction_horizon_days": prediction_horizon_days,
        "tta": f"ON x{rounds}" if enable_tta else "OFF",
        "device": str(device),
    }


def main():
    parser = argparse.ArgumentParser(description="TR-RADIL Single-Patient Inference")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--sex", type=str, required=True, choices=["male", "female"])
    parser.add_argument("--baseline_tr_grade", type=int, required=True, choices=[1, 2])
    parser.add_argument("--prediction_horizon_days", type=int, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--is_external", action="store_true")
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--tta_rounds", type=int, default=5)
    args = parser.parse_args()

    for path, name in [(args.video_path, "Video"), (args.checkpoint_path, "Checkpoint")]:
        if not os.path.isfile(path):
            print(f"Error: {name} not found: {path}"); sys.exit(1)

    result = predict(
        args.video_path, args.age, args.sex, args.baseline_tr_grade,
        args.prediction_horizon_days, args.checkpoint_path,
        args.is_external, not args.no_tta, args.tta_rounds
    )

    grade_map = {1: "Mild", 2: "Moderate", 3: "Severe"}
    print(f"\n{'=' * 60}")
    print(f"  TR-RADIL Inference Result")
    print(f"{'=' * 60}")
    print(f"  Video              : {os.path.basename(args.video_path)}")
    print(f"  Age                : {args.age}")
    print(f"  Sex                : {args.sex}")
    print(f"  Baseline TR        : Grade {args.baseline_tr_grade} "
          f"({grade_map.get(args.baseline_tr_grade, '')})")
    print(f"  Prediction Horizon : {args.prediction_horizon_days} days "
          f"({args.prediction_horizon_days / 365:.1f} years)")
    print(f"{'-' * 60}")
    print(f"  Progression Prob   : {result['progression_probability']:.1%}")
    print(f"  Prediction         : {result['prediction']}")
    print(f"  Follow-up TR Grade : {result['predicted_followup_grade']} "
          f"({result['predicted_followup_severity']})")
    print(f"{'-' * 60}")
    print(f"  TTA                : {result['tta']}")
    print(f"  Device             : {result['device']}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
