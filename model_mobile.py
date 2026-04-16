# model_mobile.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC,
    MulticlassConfusionMatrix, MulticlassRecall, MulticlassPrecision
)
from config import Config


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation on video tensors only."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class SpatialAttention(nn.Module):
    """2-layer Conv1x1 spatial attention: (C) -> (C//8) -> (1) -> Sigmoid."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.conv1(x)
        attn = self.relu(attn)
        attn = self.conv2(attn)
        attn_map = self.sigmoid(attn)
        return x * attn_map, attn_map


class Echo_RADIL(pl.LightningModule):
    """
    TR-RADIL: Multimodal framework for TR progression prediction.

    Architecture:
        Video branch: MobileNetV3-Large (partial fine-tune) + Spatial Attention
                      + Transformer Encoder + Temporal Attention Pooling
        Clinical branch: Linear(4->32) + LayerNorm + Interval Bucket Embedding
        Fusion: Concatenation -> State Estimator -> Trajectory Modeling
        Output: Binary classifier (main) + 3-class severity (auxiliary)
    """
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        # --- Spatial backbone: MobileNetV3-Large ---
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        backbone = models.mobilenet_v3_large(weights=weights)
        self.feature_extractor = backbone.features  # Output: (B, 960, 7, 7)
        self.feat_dim = 960

        # Freeze layers 0-12, fine-tune layers 13-16
        for i, child in enumerate(self.feature_extractor.children()):
            for param in child.parameters():
                param.requires_grad = (i >= 13)

        # --- Spatial attention ---
        self.spatial_attn = SpatialAttention(self.feat_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # --- Temporal transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feat_dim, nhead=8, dim_feedforward=2048,
            dropout=Config.DROPOUT_RATE, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pos_embed = nn.Parameter(torch.randn(1, Config.NUM_FRAMES, self.feat_dim) * 0.02)

        # --- Temporal attention pooling ---
        self.temporal_attn_pool = nn.Sequential(
            nn.Linear(self.feat_dim, 64),
            nn.Tanh(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(64, 1)
        )

        # --- Clinical encoding ---
        self.clinical_proj = nn.Sequential(
            nn.Linear(4, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE)
        )

        # --- Interval bucket embedding ---
        # Boundaries at ~2yr, ~5yr, ~10yr (in normalized units: days/3650)
        self.interval_bucket_edges = torch.tensor([0.2, 0.5, 1.0], dtype=torch.float32)
        self.interval_embed = nn.Embedding(4, 8)

        # --- Current state estimator ---
        # Input: video(960) + clinical(32) + interval(8) = 1000
        self.current_state_estimator = nn.Sequential(
            nn.Linear(self.feat_dim + 32 + 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(128, Config.HIDDEN_DIM)  # -> 64-d
        )

        # --- Auxiliary classifier: predict follow-up severity grade ---
        self.aux_classifier = nn.Linear(Config.HIDDEN_DIM, Config.AUX_NUM_CLASSES)

        # --- History (baseline severity) embedding ---
        self.history_embedding = nn.Embedding(Config.NUM_SEVERITY_LEVELS, Config.HIDDEN_DIM)

        # --- Main classifier: trajectory features -> binary prediction ---
        # Input: [current; history; diff; product] = 64*4 = 256
        self.classifier = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM * 4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(64, Config.NUM_CLASSES)
        )

        # --- Loss functions ---
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32),
            label_smoothing=Config.LABEL_SMOOTHING
        )
        self.aux_criterion = nn.CrossEntropyLoss()

        # --- Metrics ---
        metrics = MetricCollection({
            'acc': MulticlassAccuracy(num_classes=Config.NUM_CLASSES),
            'f1_weighted': MulticlassF1Score(num_classes=Config.NUM_CLASSES, average='weighted'),
            'auc': MulticlassAUROC(num_classes=Config.NUM_CLASSES, average='weighted'),
            'recall_weighted': MulticlassRecall(num_classes=Config.NUM_CLASSES, average='weighted'),
            'precision_weighted': MulticlassPrecision(num_classes=Config.NUM_CLASSES, average='weighted')
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.val_cm = MulticlassConfusionMatrix(num_classes=Config.NUM_CLASSES)

    def forward(self, video, clinical):
        b, c, t, h, w = video.shape
        x = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        # Spatial feature extraction
        feat_map = self.feature_extractor(x)
        feat_map_weighted, attn_map = self.spatial_attn(feat_map)
        feat_vec = self.avg_pool(feat_map_weighted).flatten(1)  # (B*T, 960)

        # Temporal modeling
        feat_seq = feat_vec.view(b, t, -1)
        feat_seq = feat_seq + self.pos_embed[:, :t, :]
        feat_trans = self.temporal_transformer(feat_seq)

        # Temporal attention pooling
        t_attn_scores = self.temporal_attn_pool(feat_trans)
        t_attn_weights = F.softmax(t_attn_scores, dim=1)
        video_feat = torch.sum(feat_trans * t_attn_weights, dim=1)  # (B, 960)

        # Clinical encoding
        clin_feat = self.clinical_proj(clinical)  # (B, 32)

        # Interval bucket embedding
        interval_norm = clinical[:, 3].clamp(min=0)
        bucket_edges = self.interval_bucket_edges.to(interval_norm.device)
        bucket_idx = torch.bucketize(interval_norm, bucket_edges)
        interval_emb = self.interval_embed(bucket_idx)  # (B, 8)

        # Multimodal fusion
        combined = torch.cat([video_feat, clin_feat, interval_emb], dim=1)  # (B, 1000)

        # Current state estimation
        current_state = self.current_state_estimator(combined)  # (B, 64)
        aux_logits = self.aux_classifier(current_state)

        # History state from baseline severity
        first_val_norm = clinical[:, 0]
        first_val_idx = torch.round(
            first_val_norm * (Config.NUM_SEVERITY_LEVELS - 1)
        ).long().clamp(0, Config.NUM_SEVERITY_LEVELS - 1)
        history_state = self.history_embedding(first_val_idx)  # (B, 64)

        # Trajectory features
        diff_feat = current_state - history_state
        prod_feat = current_state * history_state
        final_input = torch.cat([current_state, history_state, diff_feat, prod_feat], dim=1)

        logits = self.classifier(final_input)
        return logits, aux_logits, attn_map, t_attn_weights

    def training_step(self, batch, batch_idx):
        video, clinical, label, aux_label = batch

        # Mixup augmentation (video tensors only, p=0.3)
        if self.training and torch.rand(1).item() < 0.3:
            video, label_a, label_b, lam = mixup_data(video, label, alpha=0.2)
            logits, aux_logits, _, _ = self(video, clinical)
            loss_main = lam * self.criterion(logits, label_a) + (1 - lam) * self.criterion(logits, label_b)
            loss_aux = self.aux_criterion(aux_logits, aux_label)
            loss = loss_main + Config.AUX_LOSS_WEIGHT * loss_aux
            self.train_metrics(F.softmax(logits, dim=1), label_a)
        else:
            logits, aux_logits, _, _ = self(video, clinical)
            loss_main = self.criterion(logits, label)
            loss_aux = self.aux_criterion(aux_logits, aux_label)
            loss = loss_main + Config.AUX_LOSS_WEIGHT * loss_aux
            self.train_metrics(F.softmax(logits, dim=1), label)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        video, clinical, label, aux_label = batch
        logits, aux_logits, _, _ = self(video, clinical)
        loss_main = self.criterion(logits, label)
        loss_aux = self.aux_criterion(aux_logits, aux_label)
        loss = loss_main + Config.AUX_LOSS_WEIGHT * loss_aux

        probs = F.softmax(logits, dim=1)
        self.val_metrics(probs, label)
        self.val_cm.update(torch.argmax(logits, dim=1), label)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            print(f"\nConfusion Matrix:\n{self.val_cm.compute().cpu().numpy()}")
        self.val_cm.reset()

    def configure_optimizers(self):
        backbone_params = []
        head_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "feature_extractor" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': Config.LR_BACKBONE},
            {'params': head_params, 'lr': Config.LR_HEAD}
        ], weight_decay=Config.WEIGHT_DECAY)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=8, T_mult=2, eta_min=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }
