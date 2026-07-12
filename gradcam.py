"""Generate Grad-CAM visualizations for a single patient.

Usage:
    python gradcam.py \
        --video_path data/patient_001.nii.gz \
        --age 65 --sex female --baseline_tr_grade 2 \
        --prediction_horizon_days 365 \
        --checkpoint_path weights/best_model.ckpt \
        --output_dir gradcam आउट
"""
import argparse
import os

import cv2
import numpy as np
import torch

from inference import load_video, preprocess_frames, encode_clinical
from model_mobile import Echo_RADIL


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize_frame(frame_t: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor frame (3, H, W) back to uint8 RGB."""
    frame = frame_t.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    frame = frame * STD + MEAN
    frame = np.clip(frame, 0.0, 1.0)
    return (frame * 255.0).astype(np.uint8)


def overlay_cam(frame_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = (cam * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, heat, alpha, 0.0)


def build_montage(overlay_frames, cols=4):
    if not overlay_frames:
        raise ValueError("No frames to assemble")
    rows = int(np.ceil(len(overlay_frames) / cols))
    h, w = overlay_frames[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(overlay_frames):
        r, c = divmod(idx, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    return canvas


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for TR-RADIL")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--sex", type=str, required=True, choices=["male", "female"])
    parser.add_argument("--baseline_tr_grade", type=int, required=True, choices=[1, 2])
    parser.add_argument("--prediction_horizon_days", type=int, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./gradcam_outputs")
    parser.add_argument("--target_class", type=int, default=-1, help="-1 uses predicted class")
    parser.add_argument("--is_external", action="store_true")
    parser.add_argument("--max_frames", type=int, default=8)
    args = parser.parse_args()

    for path, name in [(args.video_path, "Video"), (args.checkpoint_path, "Checkpoint")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Echo_RADIL.load_from_checkpoint(args.checkpoint_path, map_location=device)
    model.to(device).eval()

    video_raw = load_video(args.video_path)
    video_t = preprocess_frames(video_raw, args.is_external).unsqueeze(0).to(device)
    clinical_t = encode_clinical(
        args.age, args.sex, args.baseline_tr_grade, args.prediction_horizon_days
    ).unsqueeze(0).to(device)

    target = None if args.target_class < 0 else args.target_class
    result = model.generate_gradcam(video_t, clinical_t, target_class=target)

    cams = result["gradcam"][0].cpu().numpy()
    pred_class = int(result["pred_class"][0].item())
    logits = result["logits"][0].cpu().numpy()
    prob = torch.softmax(result["logits"], dim=1)[0, 1].item()

    frame_scores = cams.reshape(cams.shape[0], -1).mean(axis=1)
    top_indices = np.argsort(frame_scores)[::-1][: max(1, min(args.max_frames, cams.shape[0]))]

    overlays = []
    for idx in top_indices:
        frame_rgb = denormalize_frame(video_t[0, :, idx])
        overlays.append(overlay_cam(frame_rgb, cams[idx]))

    montage = build_montage(overlays, cols=min(4, len(overlays)))
    out_img = os.path.join(args.output_dir, "gradcam_montage.png")
    out_npy = os.path.join(args.output_dir, "gradcam_heatmap.npy")
    out_txt = os.path.join(args.output_dir, "gradcam_result.txt")

    cv2.imwrite(out_img, montage)
    np.save(out_npy, cams)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"pred_class={pred_class}\n")
        f.write(f"worsen_probability={prob:.6f}\n")
        f.write(f"logits={logits.tolist()}\n")
        f.write(f"saved_montage={out_img}\n")
        f.write(f"saved_heatmap={out_npy}\n")

    print(f"Saved Grad-CAM montage to: {out_img}")
    print(f"Saved heatmap tensor to: {out_npy}")
    print(f"Predicted worsen probability: {prob:.4f}")


if __name__ == "__main__":
    main()
