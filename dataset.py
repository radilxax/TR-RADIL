# dataset_WithRM_corner.py
import pandas as pd
import numpy as np
import torch
import nibabel as nib
from typing import Any, Dict, Optional, cast
import cv2
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from config import Config

class CardiacMultiModalDataset(Dataset):
    def __init__(self, csv_path, mode='train'):
        self.mode = mode 
        self.is_empty = False
        
        if csv_path.endswith('.xlsx') or csv_path.endswith('.xls'):
            df = pd.read_excel(csv_path)
        else:
            df = pd.read_csv(csv_path)
                
        def map_to_2class(diff):
            if diff > 0:
                return 1 # Worsen
            else:
                return 0 # Stable + Improve
        
        if Config.COL_SEVERITY_DIFF in df.columns:
            df[Config.COL_LABEL] = df[Config.COL_SEVERITY_DIFF].apply(map_to_2class)
        else:
            raise ValueError(f"❌ CSV中缺少 '{Config.COL_SEVERITY_DIFF}' 列！")
        
        self.df = df
        
        if len(self.df) > 0:
            label_dist = self.df[Config.COL_LABEL].value_counts().sort_index()
            mode_name = {'train': '训练集', 'val': '验证集', 'test': '测试集'}.get(mode, mode)
            print(f"\n[{mode_name}] 样本数: {len(self.df)}")
            print(f"  Improve + stable (0): {label_dist.get(0, 0)}")
            print(f"  Worsen  (1): {label_dist.get(1, 0)}")
        else:
            print(f"\n{mode} 数据集为空: {csv_path}")
            self.is_empty = True
                

        self.normalize = A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0, 
            p=1.0
        )

        # if mode == 'train':
        #     self.aug = A.ReplayCompose([
        #         A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.7),
                
        #         # 1. 安全几何变换
        #         A.Affine(
        #             translate_percent=0.0625,
        #             scale=(0.8, 1.1),
        #             rotate=(-15, 15),
        #             border_mode=cv2.BORDER_CONSTANT,
        #             fill=0,
        #             p=0.7
        #         ),
                
        #         # 2. 形状扭曲
        #         A.OneOf([
        #             A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        #             A.ElasticTransform(alpha=1, sigma=50, p=0.5), 
        #         ], p=0.5),

        #         # 3. 噪声与模糊
        #         A.OneOf([
        #             # std_range 归一化到 [0,1]
        #             A.GaussNoise(std_range=(3.0 / 255.0, 7.0 / 255.0), p=0.5),
        #             A.MotionBlur(blur_limit=5, p=0.5),
        #             A.ISONoise(p=0.5)
        #         ], p=0.5),
                
        #         A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                
        #         # 4. 遮挡
        #         A.CoarseDropout(
        #             num_holes_range=(4, 10),
        #             hole_height_range=(10, 25),
        #             hole_width_range=(10, 25),
        #             fill=0, 
        #             p=0.5
        #         ),
                
        #         self.normalize, 
        #         ToTensorV2()   
        #     ])
        # else:
        #     self.aug = A.Compose([
        #         A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        #         self.normalize,
        #         ToTensorV2()
        #     ])
        if mode == 'train':
            self.aug = A.ReplayCompose([
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.7),
                
                A.Affine(
                    translate_percent=0.05,
                    scale=(0.9, 1.1),
                    rotate=(-10, 10),
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    p=0.5
                ),
                
                A.OneOf([
                    A.GaussNoise(std_range=(2.0/255.0, 5.0/255.0), p=0.5),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                ], p=0.3),
                
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                
                A.CoarseDropout(
                    num_holes_range=(2, 6),
                    hole_height_range=(8, 20),
                    hole_width_range=(8, 20),
                    fill=0,
                    p=0.3
                ),
                
                self.normalize,
                ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                self.normalize,
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)
    
    # def _load_video(self, full_path):
    #     try:
    #         nii = cast(Any, nib.load(full_path))
            
    #         header_dtype = nii.header.get_data_dtype()
            
    #         if header_dtype.names is not None and 'R' in header_dtype.names:
    #             # RGB 格式：dtype = [('R','u1'), ('G','u1'), ('B','u1')]
    #             raw = np.asanyarray(nii.dataobj)  # 不用 get_fdata
    #             # 提取 R/G/B 通道并转灰度
    #             r = raw['R'].astype(np.float32)
    #             g = raw['G'].astype(np.float32)
    #             b = raw['B'].astype(np.float32)
    #             v = 0.299 * r + 0.587 * g + 0.114 * b
    #         else:
    #             # 普通格式
    #             v = nii.get_fdata(dtype=np.float32)
            
    #         v = np.squeeze(v)
            
    #         # 维度修正 (H, W, T) -> (T, H, W)
    #         if v.ndim == 4 and v.shape[-1] == 3:
    #             v = 0.299 * v[..., 0] + 0.587 * v[..., 1] + 0.114 * v[..., 2]

    #         if v.ndim == 2:
    #             v = v[np.newaxis, :, :]
    #         elif v.ndim == 3:
    #             v = np.transpose(v, (2, 0, 1))
    #         v = np.ascontiguousarray(v)
            
    #         T_raw = v.shape[0]
            
    #         if T_raw >= Config.NUM_FRAMES:
    #             diffs = np.zeros(T_raw)
    #             for t in range(1, T_raw):
    #                 diffs[t] = np.mean(np.abs(v[t].astype(np.float32) - v[t-1].astype(np.float32)))
                
    #             n_motion = Config.NUM_FRAMES // 2
    #             n_uniform = Config.NUM_FRAMES - n_motion
                
    #             motion_indices = np.argsort(diffs)[::-1][:n_motion]
    #             uniform_indices = np.linspace(0, T_raw - 1, n_uniform).astype(int)
                
    #             all_indices = np.unique(np.concatenate([motion_indices, uniform_indices]))
                
    #             if len(all_indices) > Config.NUM_FRAMES:
    #                 sub_idx = np.linspace(0, len(all_indices) - 1, Config.NUM_FRAMES).astype(int)
    #                 all_indices = all_indices[sub_idx]
    #             elif len(all_indices) < Config.NUM_FRAMES:
    #                 pad_num = Config.NUM_FRAMES - len(all_indices)
    #                 all_indices = np.concatenate([all_indices, np.full(pad_num, all_indices[-1])])
                
    #             v = v[all_indices]
    #         else:
    #             pad_num = Config.NUM_FRAMES - T_raw
    #             padding = np.tile(v[-1:], (pad_num, 1, 1))
    #             v = np.concatenate([v, padding], axis=0)
            
    #         return v
            
    #     except Exception as e:
    #         raise RuntimeError(f"Load Error: {e}")
    def _load_video(self, full_path):
        try:
            nii = cast(Any, nib.load(full_path))
            
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
                v = np.transpose(v, (2, 0, 1))
            v = np.ascontiguousarray(v)
            
            T_raw = v.shape[0]
            
            if T_raw >= Config.NUM_FRAMES:
                if self.mode == 'train':
                    max_start = T_raw - Config.NUM_FRAMES
                    start = np.random.randint(0, max_start + 1)
                    v = v[start : start + Config.NUM_FRAMES]
                else:
                    start = (T_raw - Config.NUM_FRAMES) // 2
                    v = v[start : start + Config.NUM_FRAMES]
            else:
                pad_num = Config.NUM_FRAMES - T_raw
                padding = np.tile(v[-1:], (pad_num, 1, 1))
                v = np.concatenate([v, padding], axis=0)
            
            return v
            
        except Exception as e:
            print(f"❌ 加载视频失败 {full_path}: {e}")
            raise RuntimeError(f"Load Error: {e}")

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        
        if self.is_empty: return self._get_dummy_sample()
        
        row = self.df.iloc[idx]
        full_path = row[Config.COL_PATH]
        
        try:
            label = int(row[Config.COL_LABEL])
            video_raw = self._load_video(full_path)
            
            frames = []
            replay: Optional[Dict[str, Any]] = None
            
            global_min, global_max = video_raw.min(), video_raw.max()
            
            for i in range(video_raw.shape[0]):
                fr = video_raw[i]
                if self.mode == 'test':
                    fr = np.rot90(fr, k=1)          
                    fr = np.flip(fr, axis=0)         
                    fr = np.ascontiguousarray(fr)
                # 1. Center Crop (80%)
                h_raw, w_raw = fr.shape[:2]
                crop_ratio = 0.7
                h_center, w_center = h_raw // 2, w_raw // 2
                h_crop, w_crop = int(h_raw * crop_ratio) // 2, int(w_raw * crop_ratio) // 2
                
                if h_crop > 0 and w_crop > 0:
                    fr = fr[h_center - h_crop : h_center + h_crop, 
                            w_center - w_crop : w_center + w_crop]
                
                # 2. Resize
                fr = cv2.resize(fr, (Config.IMG_SIZE, Config.IMG_SIZE))
                
                # 3. 归一化转 uint8
                if fr.dtype != np.uint8:
                    if global_max > global_min:
                        fr = 255.0 * (fr - global_min) / (global_max - global_min)
                    else:
                        fr = np.zeros_like(fr)
                    fr = fr.astype(np.uint8)
                
                # 4. 转 RGB
                fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2RGB)

                h, w = fr.shape[:2]
                
                left_strip = w // 10
                fr[:, :left_strip] = 0
                
                pad = int(0.03 * min(h, w))
                if pad > 0:
                    fr[:pad, :] = 0   # Top
                    fr[-pad:, :] = 0  # Bottom
                    fr[:, -pad:] = 0  # Right (Left 已经被遮挡了)

                if self.mode == 'train':
                    if i == 0:
                        augmented = self.aug(image=fr)
                        replay = augmented['replay']
                        frame_aug = augmented['image']
                    else:
                        assert replay is not None
                        frame_aug = A.ReplayCompose.replay(replay, image=fr)['image']
                else:
                    frame_aug = self.aug(image=fr)['image']
                frames.append(frame_aug)
            
            video_t = torch.stack(frames).permute(1, 0, 2, 3) 
            
            first_val_raw = float(row[Config.COL_FIRST_VAL])
            first_val = (first_val_raw - 1.0) / (Config.NUM_SEVERITY_LEVELS - 1.0)  # 1->0.0, 2->0.5, 3->1.0
            
            age = float(row[Config.COL_AGE]) / 100.0
            gender = 1.0 if str(row[Config.COL_GENDER]).strip() == '女' else 0.0
            interval = float(row[Config.COL_TIME_INTERVAL]) / 3650.0
            clinical_t = torch.tensor([first_val, age, gender, interval], dtype=torch.float32)

            last_val_raw = int(row[Config.COL_LAST_VAL])
            last_val = last_val_raw - 1  # 1->0, 2->1, 3->2
            last_val = int(np.clip(last_val, 0, Config.AUX_NUM_CLASSES - 1))
            aux_label = torch.tensor(last_val, dtype=torch.long)
            
            return video_t, clinical_t, torch.tensor(label, dtype=torch.long), aux_label
            
        except Exception:
            return self._get_dummy_sample()

    def _get_dummy_sample(self):
        video_t = torch.zeros((3, Config.NUM_FRAMES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=torch.float32)
        clinical_t = torch.zeros(4, dtype=torch.float32)
        label = torch.tensor(0, dtype=torch.long)
        aux_label = torch.tensor(1, dtype=torch.long)  # ✅ 3分类的中间值 (moderate)
        return video_t, clinical_t, label, aux_label

class CardiacDataModule(pl.LightningDataModule):
    def setup(self, stage=None):
        self.train_ds = CardiacMultiModalDataset(Config.TRAIN_CSV, mode='train')
        self.val_ds = CardiacMultiModalDataset(Config.VAL_CSV, mode='val')

        if not self.train_ds.is_empty:
            labels = self.train_ds.df[Config.COL_LABEL].values

            if Config.CLASS_WEIGHTS is not None:
                class_weights = np.array(Config.CLASS_WEIGHTS, dtype=np.float32)
            else:
                class_counts = np.bincount(labels, minlength=Config.NUM_CLASSES)
                class_weights = 1.0 / (class_counts + 1e-6)

            sample_weights = class_weights[labels]

            # self.train_sampler = WeightedRandomSampler(
            #     weights=sample_weights,
            #     num_samples=len(sample_weights),
            #     replacement=True
            # )
            self.train_sampler = None
        else:
            self.train_sampler = None

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=Config.BATCH_SIZE, 
            sampler=self.train_sampler,
            shuffle=True, 
            num_workers=Config.NUM_WORKERS,
            pin_memory=True, 
            drop_last=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True
        )