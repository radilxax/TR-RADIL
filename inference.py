"""
TR-RADIL Single-Patient Inference Script

Predicts tricuspid regurgitation (TR) progression risk from a baseline
A4C echocardiographic video and minimal clinical variables.

Usage:
    python inference.py \
        --video_path data/patient_001.nii.gz \
        --age 65 --sex female --baseline_tr_grade 2 \
        --prediction_horizon_days 365 \
        --checkpoint_path weights/best_model.ckpt

Inputs:
    - video_path              : A4C echocardiographic video (.nii.gz)
    - age                     : Patient age in years
    - sex                     : male or female
    - baseline_tr_grade       : 1 (mild) or 2 (moderate)
    - prediction_horizon_days : Target prediction window in days
    - checkpoint_path         : Path to trained model checkpoint (.ckpt)

Outputs:
    - Predicted probability of TR progression (0-1)
    - Binary prediction (Worsen / Stable+Improved)
    - Predicted follow-up TR severity grade (1=mild, 2=moderate, 3=severe)
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


# ---------------------------------------------------------------------------
# Video loading & preprocessing (mirrors dataset.py for val/test mode)
# ---------------------------------------------------------------------------

def load_video(video_path: str) -> np.ndarray:
    """Load NIfTI video and return array of shape (T, H, W) in float32."""
    nii = cast(Any, nib.load(video_path))

    header_dtype = nii.header.get_data_dtype()
    if header_dtype.names is not None and 'R' in header_dtype.names:
        raw = np.asanyarray(nii.dataobj)
        r = raw['R'].astype(np.float32)
        g = raw['G'].astype(np.float32)
        b = raw['B'].astype(np.float32)
        v = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        v = nii.get_fdata(dtype=np.float32)

    v = np.squeeze(v)

    if v.ndim == 4 and v.shape[-1] == 3:
        v = 0.299 * v[..., 0] + 0.587 * v[..., 1] + 0.114 * v[..., 2]
    if v.ndim == 2:
        v = v[np.newaxis, :, :]
    elif v.ndim == 3:
        v = np.transpose(v, (2, 0, 1))  # (H, W, T) -> (T, H, W)

    v = np.ascontiguousarray(v)

    # Temporal sampling: center crop to NUM_FRAMES
    T_raw = v.shape[0]
    if T_raw >= Config.NUM_FRAMES:
        start = (T_raw - Config.NUM_FRAMES) // 2
        v = v[start: start + Config.NUM_FRAMES]
    else:
        pad_num = Config.NUM_FRAMES - T_raw
        padding = np.tile(v[-1:], (pad_num, 1, 1))
        v = np.concatenate([v, padding], axis=0)

    return v


def preprocess_frames(video_raw: np.ndarray, is_external: bool = False) -> torch.Tensor:
    """
    Per-frame spatial preprocessing identical to dataset.py (val/test mode).

    Returns: (3, T, H, W) float32 tensor, ImageNet-normalised.
    """
    normalize = A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
        p=1.0,
    )
    aug = A.Compose([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        normalize,
        ToTensorV2(),
    ])

    global_min, global_max = video_raw.min(), video_raw.max()
    frames = []

    for i in range(video_raw.shape[0]):
        fr = video_raw[i]

        # Orientation correction for external-center data
        if is_external:
            fr = np.rot90(fr, k=1)
            fr = np.flip(fr, axis=0)
            fr = np.ascontiguousarray(fr)

        # 1. Center crop (keep 70%)
        h_raw, w_raw = fr.shape[:2]
        crop_ratio = 0.7
        h_center, w_center = h_raw // 2, w_raw // 2
        h_crop = int(h_raw * crop_ratio) // 2
        w_crop = int(w_raw * crop_ratio) // 2
        if h_crop > 0 and w_crop > 0:
            fr = fr[h_center - h_crop: h_center + h_crop,
                     w_center - w_crop: w_center + w_crop]

        # 2. Resize to 224x224
        fr = cv2.resize(fr, (Config.IMG_SIZE, Config.IMG_SIZE))

        # 3. Min-max -> uint8
        if fr.dtype != np.uint8:
            if global_max > global_min:
                fr = 255.0 * (fr - global_min) / (global_max - global_min)
            else:
                fr = np.zeros_like(fr)
            fr = fr.astype(np.uint8)

        # 4. Gray -> RGB
        fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2RGB)

        # 5. Edge masking
        h, w = fr.shape[:2]
        left_strip = w // 10
        fr[:, :left_strip] = 0
        pad = int(0.03 * min(h, w))
        if pad > 0:
            fr[:pad, :] = 0
            fr[-pad:, :] = 0
            fr[:, -pad:] = 0

        # 6. CLAHE + ImageNet normalise + to tensor
        frame_aug = aug(image=fr)['image']
        frames.append(frame_aug)

    video_t = torch.stack(frames).permute(1, 0, 2, 3)
    return video_t


def encode_clinical(
    age: float,
    sex: str,
    baseline_tr_grade: int,
    prediction_horizon_days: int,
) -> torch.Tensor:
    """
    Encode clinical variables into a 4-d tensor matching training format.
        [0] baseline_tr_severity : (grade - 1) / 2
        [1] age                  : age / 100
        [2] sex                  : female=1.0, male=0.0
        [3] follow_up_interval   : days / 3650
    """
    first_val = (baseline_tr_grade - 1.0) / (Config.NUM_SEVERITY_LEVELS - 1.0)
    age_norm = age / 100.0
    gender = 1.0 if sex.lower() in ('female', 'f') else 0.0
    interval = prediction_horizon_days / 3650.0
    return torch.tensor([first_val, age_norm, gender, interval], dtype=torch.float32)


# ---------------------------------------------------------------------------
# TTA (test-time augmentation) — mirrors test.py
# ---------------------------------------------------------------------------

def apply_tta(video: torch.Tensor, tta_idx: int) -> torch.Tensor:
    if tta_idx == 0:
        return video
    elif tta_idx == 1:
        return torch.flip(video, dims=[-1])       # horizontal flip
    elif tta_idx == 2:
        return torch.flip(video, dims=[-2])       # vertical flip
    elif tta_idx == 3:
        return video * 1.05                       # brightness +5%
    else:
        return video * 0.95                       # brightness -5%


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_single_patient(
    video_path: str,
    age: float,
    sex: str,
    baseline_tr_grade: int,
    prediction_horizon_days: int,
    checkpoint_path: str,
    is_external: bool = False,
    enable_tta: bool = True,
    tta_rounds: int = 5,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    model = Echo_RADIL.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    # Preprocess video
    video_raw = load_video(video_path)
    video_tensor = preprocess_frames(video_raw, is_external=is_external)
    video_tensor = video_tensor.unsqueeze(0).to(device)  # (1, 3, T, H, W)

    # Encode clinical variables
    clinical_tensor = encode_clinical(
        age, sex, baseline_tr_grade, prediction_horizon_days
    ).unsqueeze(0).to(device)

    # Inference with optional TTA
    rounds = tta_rounds if enable_tta else 1
    accumulated_probs = None

    for tta_idx in range(rounds):
        v_input = apply_tta(video_tensor, tta_idx)
        logits, aux_logits, _, _ = model(v_input, clinical_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()

        if accumulated_probs is None:
            accumulated_probs = probs
        else:
            accumulated_probs += probs

    final_probs = accumulated_probs / rounds
    prob_worsen = float(final_probs[0, 1])
    pred_class = int(np.argmax(final_probs[0]))

    # Auxiliary: predicted follow-up TR severity (from non-augmented pass)
    logits_clean, aux_logits_clean, _, _ = model(video_tensor, clinical_tensor)
    aux_pred = int(torch.argmax(aux_logits_clean, dim=1).item()) + 1

    label_map = {0: "Stable/Improved", 1: "Worsen"}
    grade_map = {1: "Mild", 2: "Moderate", 3: "Severe"}

    return {
        "video_path": os.path.basename(video_path),
        "patient_info": {
            "age": age,
            "sex": sex,
            "baseline_tr_grade": baseline_tr_grade,
            "baseline_tr_severity": grade_map.get(baseline_tr_grade, str(baseline_tr_grade)),
        },
        "prediction_horizon_days": prediction_horizon_days,
        "progression_probability": round(prob_worsen, 4),
        "prediction": label_map[pred_class],
        "predicted_followup_grade": aux_pred,
        "predicted_followup_severity": grade_map.get(aux_pred, str(aux_pred)),
        "tta_enabled": enable_tta,
        "tta_rounds": rounds,
        "device": str(device),
    }


def print_result(result: dict):
    print("\n" + "=" * 60)
    print("  TR-RADIL Inference Result")
    print("=" * 60)
    print(f"  Video              : {result['video_path']}")
    print(f"  Age                : {result['patient_info']['age']}")
    print(f"  Sex                : {result['patient_info']['sex']}")
    print(f"  Baseline TR        : Grade {result['patient_info']['baseline_tr_grade']} "
          f"({result['patient_info']['baseline_tr_severity']})")
    print(f"  Prediction Horizon : {result['prediction_horizon_days']} days "
          f"({result['prediction_horizon_days'] / 365:.1f} years)")
    print("-" * 60)
    print(f"  Progression Prob   : {result['progression_probability']:.1%}")
    print(f"  Prediction         : {result['prediction']}")
    print(f"  Follow-up TR Grade : {result['predicted_followup_grade']} "
          f"({result['predicted_followup_severity']})")
    print("-" * 60)
    print(f"  TTA                : {'ON x' + str(result['tta_rounds']) if result['tta_enabled'] else 'OFF'}")
    print(f"  Device             : {result['device']}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="TR-RADIL: Single-patient TR progression prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1-year risk prediction
  python inference.py --video_path patient.nii.gz --age 65 --sex female \\
      --baseline_tr_grade 2 --prediction_horizon_days 365 \\
      --checkpoint_path weights/best_model.ckpt

  # 3-year risk, no TTA (faster)
  python inference.py --video_path patient.nii.gz --age 72 --sex male \\
      --baseline_tr_grade 1 --prediction_horizon_days 1095 \\
      --checkpoint_path weights/best_model.ckpt --no_tta
        """,
    )

    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to A4C echocardiographic video (.nii.gz)")
    parser.add_argument("--age", type=float, required=True,
                        help="Patient age in years")
    parser.add_argument("--sex", type=str, required=True, choices=["male", "female"],
                        help="Patient sex")
    parser.add_argument("--baseline_tr_grade", type=int, required=True, choices=[1, 2],
                        help="Baseline TR severity: 1 (mild) or 2 (moderate)")
    parser.add_argument("--prediction_horizon_days", type=int, required=True,
                        help="Target prediction window in days (e.g., 365, 730, 1095)")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to trained model checkpoint (.ckpt)")
    parser.add_argument("--is_external", action="store_true", default=False,
                        help="Apply orientation correction for external-center data")
    parser.add_argument("--no_tta", action="store_true", default=False,
                        help="Disable test-time augmentation")
    parser.add_argument("--tta_rounds", type=int, default=5,
                        help="Number of TTA rounds (default: 5)")

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        print(f"Error: Video file not found: {args.video_path}"); sys.exit(1)
    if not os.path.isfile(args.checkpoint_path):
        print(f"Error: Checkpoint not found: {args.checkpoint_path}"); sys.exit(1)

    result = predict_single_patient(
        video_path=args.video_path,
        age=args.age,
        sex=args.sex,
        baseline_tr_grade=args.baseline_tr_grade,
        prediction_horizon_days=args.prediction_horizon_days,
        checkpoint_path=args.checkpoint_path,
        is_external=args.is_external,
        enable_tta=not args.no_tta,
        tta_rounds=args.tta_rounds,
    )

    print_result(result)


if __name__ == "__main__":
    main()
