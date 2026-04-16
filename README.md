# TR-RADIL

**Tricuspid Regurgitation Risk Assessment via Dynamic Integration Learning**

A multimodal deep learning framework for predicting tricuspid regurgitation (TR) progression from baseline echocardiographic videos and clinical data.

> **Paper**: *A Multimodal Deep Learning Model for Risk Stratification of Tricuspid Regurgitation Progression*
> Submitted to European Heart Journal - Cardiovascular Imaging

##概述

TR-RADIL is an end-to-end framework that integrates:
- **Spatial Feature Extraction**: MobileNetV3-Large backbone (partially fine-tuned) with learnable 2-layer convolutional spatial attention
- **Temporal Modeling**: 1-layer Transformer encoder (8 heads) with learnable positional embeddings and attention-weighted temporal pooling
- **Clinical Data Fusion**: Structured clinical variables (age, sex, baseline TR severity, follow-up interval) with temporal bucket embeddings
- **Disease Trajectory Modeling**: Interaction features between estimated current state and historical baseline severity (concatenation, difference, element-wise product)
- **Multi-task Learning**: Binary progression prediction (main) + follow-up severity grade estimation (auxiliary)

## Model Architecture

```
Input: A4C Echo Video (NIfTI) + Clinical Variables
  │
  ├─ Video Branch
  │   ├─ Preprocessing: RGB→Gray, Center Crop (70%), Resize 224×224, Edge Masking
  │   ├─ Per-frame: MobileNetV3-Large (layers 0-12 frozen, 13-16 fine-tuned)
  │   ├─ Spatial Attention: Conv1×1 (960→120→1) + Sigmoid
  │   ├─ Global Average Pooling → 960-d frame vectors
  │   ├─ Positional Embeddings + Transformer Encoder (1 layer, 8 heads)
  │   └─ Temporal Attention Pooling (960→64→1, Softmax) → 960-d video feature
  │
  ├─ Clinical Branch
  │   ├─ Linear(4→32) + LayerNorm + ReLU → 32-d
  │   └─ Interval Bucket Embedding(4,8) → 8-d
  │
  ├─ Fusion: Concatenation [960 + 32 + 8] = 1000-d
  │
  ├─ State Estimation: Linear(1000→128→64) → current state (64-d)
  ├─ History Embedding: Embedding(3, 64) → baseline state (64-d)
  │
  ├─ Trajectory Features: [current; history; current−history; current⊙history] = 256-d
  │
  ├─ Main Classifier: Linear(256→64→2) → TR Progression (yes/no)
  └─ Auxiliary Classifier: Linear(64→3) → Follow-up TR Grade
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 12.0+

```bash
pip install -r requirements.txt
```

## Project Structure

```
TR-RADIL/
├── config.py              # Hyperparameters and configuration
├── dataset.py             # Multimodal dataset with video preprocessing pipeline
├── model_mobile.py        # TR-RADIL model definition
├── model_ablation.py      # Ablation study model (toggleable components)
├── train.py               # Training script
├── train_ablation.py      # Ablation study training with CLI flags
├── test.py                # Batch evaluation on external test set
├── inference.py           # Single-patient inference script
├── requirements.txt       # Python dependencies
└── LICENSE                # License
```

## Model Weights

Download the pretrained model checkpoint from [GitHub Releases](https://github.com/radilxax/TR-RADIL/releases/tag/v1.0) and place it in the `weights/` directory:

```bash
mkdir -p weights
# Download best_model.ckpt from the Releases page and move it here
mv ~/Downloads/best_model.ckpt weights/
```

## Usage

```bash
python train.py
```

### Ablation Study

```bash
# Full model (baseline)
python train_ablation.py --exp_name full_model

# Remove temporal transformer
python train_ablation.py --exp_name no_transformer --no_temporal_transformer

# Remove clinical features
python train_ablation.py --exp_name no_clinical --no_clinical

# Video-only (remove all clinical components)
python train_ablation.py --exp_name video_only --no_clinical --no_interval_embed
```

### Batch Evaluation (External Test Set)

```bash
python test.py
```

### Single-Patient Inference

#### Input Requirements

| Input |描述| Format |
|-------|-------------|--------|
| A4C Video | Baseline apical 4-chamber echocardiographic video | NIfTI (.nii.gz) |
| Age | Patient age | Years (integer) |
| Sex | Patient sex | `male` / `female` |
| Baseline TR Grade | Baseline TR severity | `1` (mild) or `2` (moderate) |
| Prediction Horizon | Target prediction time window | Days (e.g., 365) |

#### Output

| Output |描述|
|--------|-------------|
| Progression Probability | Predicted probability of TR worsening (0–1) |
| Binary Prediction | `Worsen` / `Stable+Improved` (threshold = 0.5) |
| Predicted Follow-up Grade | Estimated TR severity at follow-up (1/2/3) |

#### Examples

```bash
# 1-year risk prediction for a 65-year-old female with moderate TR
python inference.py \
    --video_path data/patient_001.nii.gz \
    --age 65 --sex female \
    --baseline_tr_grade 2 \
    --prediction_horizon_days 365 \
    --checkpoint_path weights/best_model.ckpt

# 3-year risk prediction, no TTA (faster)
python inference.py \
    --video_path data/patient_002.nii.gz \
    --age 72 --sex male \
    --baseline_tr_grade 1 \
    --prediction_horizon_days 1095 \
    --checkpoint_path weights/best_model.ckpt \
    --no_tta
```

#### Sample Output

```
============================================================
  TR-RADIL Inference Result
============================================================
  Video              : patient_001.nii.gz
  Age                : 65.0
  Sex                : female
  Baseline TR        : Grade 2 (Moderate)
  Prediction Horizon : 365 days (1.0 years)
------------------------------------------------------------
  Progression Prob   : 73.2%
  Prediction         : Worsen
  Follow-up TR Grade : 3 (Severe)
------------------------------------------------------------
  TTA                : ON x5
  Device             : cuda:0
============================================================
```

#### Prediction Horizon

The model accepts a user-specified prediction horizon (in days) as input. During training, the actual follow-up interval between baseline and follow-up examinations was used. At inference time, users can specify a target time window (e.g., 365 days for 1-year risk, 1095 days for 3-year risk). The interval is discretized into 4 temporal buckets (boundaries at approximately 2, 5, and 10 years) via a learnable embedding layer.

## Data Availability

Due to patient privacy regulations, the echocardiographic video data and clinical records used in this study cannot be publicly shared. The dataset was collected from Zhongshan Hospital (Fudan University) and its Xiamen branch under IRB approval.

## Citation

If you find this work useful, please cite:

```bibtex
@article{tr_radil_2026,
  title={A Multimodal Deep Learning Model for Risk Stratification of Tricuspid Regurgitation Progression},
  author={Shu, Xianhong and others},
  journal={European Heart Journal - Cardiovascular Imaging},
  year={2026}
}
```

## License

This project is for academic research purposes only. See [LICENSE](LICENSE) for details.

## Contact

For questions or collaboration inquiries, please contact: radilxax@sjtu.edu.cn
