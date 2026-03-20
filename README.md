# TR-RADIL

**Tricuspid Regurgitation Risk Assessment via Dynamic Integration Learning**

A multimodal deep learning framework for predicting tricuspid regurgitation (TR) progression from baseline echocardiographic videos and clinical data.

##概述

TR-RADIL is an end-to-end framework that integrates:
- **Spatial Feature Extraction**: MobileNetV3-Large backbone with learnable spatial attention
- **Temporal Modeling**: Transformer encoder with temporal attention pooling
- **Clinical Data Fusion**: Structured clinical variables (age, gender, baseline TR severity, follow-up interval)
- **Disease Trajectory Modeling**: Interaction features between estimated current state and historical baseline

## Model Architecture

![TR-RADIL Architecture](docs/architecture.png)

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
├── config.py                 # Hyperparameters and paths
├── dataset.py                # Multimodal dataset with preprocessing pipeline
├── model_mobile.py           # TR-RADIL model definition
├── train.py                  # Training script
├── test.py                   # Evaluation script
├── train_ablation.py         # Ablation study training
└── requirements.txt          # Python dependencies
```

## Usage

###培训
```bash
python train.py
```

## Data Availability

Due to patient privacy regulations, the echocardiographic video data and clinical records used in this study cannot be publicly shared. The dataset was collected from [Zhongshan Hospital, Fudan University] with IRB approval.

## Code Availability

This repository contains the complete source code for the TR-RADIL framework. Access to the code is available upon reasonable request to the corresponding author.


## License

This project is for academic research purposes only. Commercial use is prohibited without explicit permission.

## Contact

For code access requests or questions, please contact: [radilxax@sjtu.edu.cn]
