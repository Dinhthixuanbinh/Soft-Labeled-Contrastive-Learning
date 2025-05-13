# %%writefile /kaggle/working/Soft-Labeled-Contrastive-Learning/dataset/data_generator_mmwhs_raw.py
import os
import numpy as np
import pandas as pd
from torch.utils import data
import torch
import re
from pathlib import Path
import sys
import cv2
from glob import glob
from torch.utils.data import DataLoader

# --- Add project root to Python path ---
project_root_str = "/kaggle/working/Soft-Labeled-Contrastive-Learning"
project_root = Path(project_root_str)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import config
    from utils.utils_ import load_mnmx_csv, load_raw_data_mmwhs
    from dataset.data_generator_mscmrseg import ImageProcessor 
except ImportError as e:
    print(f"Import Error in data_generator_mmwhs_raw.py: {e}")
    print(f"Current sys.path: {sys.path}")
    mscmrseg_path = project_root / "dataset" / "data_generator_mscmrseg.py"
    if not mscmrseg_path.exists():
        print(f"ERROR: Required file {mscmrseg_path} (expected to contain ImageProcessor) was not found.")
    sys.exit(1)

class DataGenerator(data.Dataset):
    def __init__(self, data_dir, raw_data_dir, modality, domain='s', fold=0, split=0, bs=1,
                 num_class=4, crop_size=224, normalization='minmax', aug_mode='simple',
                 percent=99.0, augmentation=False, seed=1234, args_config=None):
        self._modality = modality
        self._domain = domain
        self._data_dir = data_dir
        self._raw_data_dir = raw_data_dir
        self._fold = fold
        self._split = split
        self._bs = bs
        self._aug = augmentation
        self._seed = seed
        self._num_class = num_class
        self._crop_size = crop_size
        self._normalization = normalization
        self._aug_mode = aug_mode
        self._percent = int(float(percent))

        if self._normalization == 'minmax':
            try:
                self._mnmx = load_mnmx_csv(self._modality, self._percent)
            except FileNotFoundError as e:
                print(f"ERROR in DataGenerator ({self._modality}, {self._domain}): {e}")
                self._mnmx = None 
            except Exception as e_csv:
                print(f"ERROR loading or parsing CSV for {self._modality} (percent {self._percent}): {e_csv}")
                self._mnmx = None
        
        self.args_config = args_config
        current_rev_flag = getattr(self.args_config, 'rev', False)

        if self._domain == 's':
            expected_modality = 'ct' if not current_rev_flag else 'mr'
            self.patient_list = list(config.MMWHS_CT_S_TRAIN_SET if self._modality.upper() == 'CT' else config.MMWHS_MR_S_TRAIN_SET)
            if 0 <= self._split < len(config.train_extra_list) and \
               0 <= self._fold < len(config.train_extra_list[self._split]):
                extra_patients = config.train_extra_list[self._split][self._fold]
                if self._modality.upper() == "CT": self.patient_list.extend([p + 32 for p in extra_patients])
                else: self.patient_list.extend(extra_patients)
            self.folder_type = "_woGT" 
            self.label_folder_type = "_withGT" 
        elif self._domain == 't':
            expected_modality = 'mr' if not current_rev_flag else 'ct'
            self.patient_list = list(config.MMWHS_MR_S_TRAIN_SET if self._modality.upper() == 'MR' else config.MMWHS_CT_S_TRAIN_SET)
            target_fold_idx = getattr(self.args_config, 'val_num', self._fold)
            if 0 <= self._split < len(config.train_extra_list) and \
               0 <= target_fold_idx < len(config.train_extra_list[self._split]):
                extra_patients = config.train_extra_list[self._split][target_fold_idx]
                if self._modality.upper() == "CT": self.patient_list.extend([p + 32 for p in extra_patients])
                else: self.patient_list.extend(extra_patients)
            self.folder_type = "_woGT" 
            self.label_folder_type = "_withGT"
        else: # Test phase
            self.folder_type = "_withGT"
            self.label_folder_type = "_withGT"
            if 0 <= self._split < len(config.train_extra_list) and \
               0 <= self._fold < len(config.train_extra_list[self._split]):
                self.patient_list = config.train_extra_list[self._split][self._fold]
                if self._modality.upper() == "CT": self.patient_list = [p + 32 for p in self.patient_list]
            else: 
                self.patient_list = list(range(1, 21)) 
                if self._modality.upper() == "CT": self.patient_list = [p + 32 for p in self.patient_list]

        self.patient_list = sorted(list(set(self.patient_list)))
        self.base_image_folder = Path(self._raw_data_dir) / f"{self._modality.upper()}{self.folder_type}"
        self.base_label_folder = Path(self._raw_data_dir) / f"{self._modality.upper()}{self.label_folder_type}"
        
        self.image_paths = []
        for pat_id in self.patient_list:
            pat_image_paths = sorted(glob(str(self.base_image_folder / f'img{pat_id}_slice*.nii')))
            self.image_paths.extend(pat_image_paths)

        if not self.image_paths:
            print(f"WARNING: No image paths found for domain '{self._domain}', modality '{self._modality}', fold {self._fold}, split {self._split}, folder_type '{self.folder_type}'")
            print(f"Searched in: {self.base_image_folder}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        m = re.search(r'img(\d+)_slice(\d+)\.nii', os.path.basename(img_path))
        img_name_key = f"img{m.group(1)}" if m else Path(img_path).stem.split('_slice')[0]

        label_filename = os.path.basename(img_path).replace('img', 'lab').replace('_slice', '_label_slice')
        label_path = str(self.base_label_folder / label_filename)
        
        img_array, mask_array = load_raw_data_mmwhs(img_path, label_path if os.path.exists(label_path) else None)

        if self._normalization == 'minmax':
            if hasattr(self, '_mnmx') and self._mnmx is not None and img_name_key in self._mnmx.index:
                min_col_name = f'min{self._percent}'
                max_col_name = f'max{self._percent}'
                if not (min_col_name in self._mnmx.columns and max_col_name in self._mnmx.columns):
                    if self._percent == 99 and 'min99' in self._mnmx.columns and 'max99' in self._mnmx.columns:
                        min_col_name = 'min99'; max_col_name = 'max99'
                    else: 
                        raise KeyError(f"CSV for {self._modality} (percent {self._percent}) missing columns '{min_col_name}' or '{max_col_name}'. Available columns in loaded CSV: {self._mnmx.columns.tolist()}")
                
                vmin = self._mnmx.loc[img_name_key, min_col_name]
                vmax = self._mnmx.loc[img_name_key, max_col_name]
                img_array = np.clip((np.array(img_array, np.float32) - vmin) / (vmax - vmin + 1e-7), 0, 1)
            else:
                lower_p = 1.0 if self._percent == 99 else (0.0 if self._percent == 100 else float(self._percent))
                upper_p = 99.0 if self._percent == 99 else (100.0 if self._percent == 100 else float(self._percent))
                
                min_val_calc = np.percentile(img_array, lower_p) 
                max_val_calc = np.percentile(img_array, upper_p)
                img_array = np.clip((img_array.astype(np.float32) - min_val_calc) / (max_val_calc - min_val_calc + 1e-7), 0, 1)

        elif self._normalization == 'zscore':
            mean = np.mean(img_array)
            std = np.std(img_array)
            img_array = (img_array.astype(np.float32) - mean) / (std + 1e-7)
        
        # Ensure img_array is 2D (H,W) before ImageProcessor methods if they expect that
        if img_array.ndim == 3 and img_array.shape[0] == 1: # (1, H, W)
            img_array = img_array.squeeze(0)
        elif img_array.ndim == 3 and img_array.shape[-1] == 1: # (H, W, 1)
            img_array = img_array.squeeze(-1)
        # Add more checks if img_array could have other 3D shapes that are actually 2D intensity data

        img_array = ImageProcessor.crop_resize(img_array, target_size=(self._crop_size, self._crop_size))
        if mask_array is not None:
            if mask_array.ndim == 3 and mask_array.shape[0] == 1: mask_array = mask_array.squeeze(0) # Ensure mask is H,W
            mask_array = ImageProcessor.crop_resize(mask_array, target_size=(self._crop_size, self._crop_size), interpolation=cv2.INTER_NEAREST)
        else: 
            mask_array = np.zeros((self._crop_size, self._crop_size), dtype=np.uint8)

        if self._aug:
            # Ensure img_array is H,W or H,W,C (if color) for aug methods; mask H,W
            if self._aug_mode == 'simple':
                img_array, mask_array = ImageProcessor.simple_aug(image=img_array, mask=mask_array) # Assumes simple_aug handles H,W
            elif self._aug_mode == 'heavy':
                 img_array, mask_array = ImageProcessor.heavy_aug(image=img_array, mask=mask_array, aug_mode=self._aug_mode)
            elif self._aug_mode == 'heavy2' and hasattr(ImageProcessor, 'heavy_aug2'): # Ensure heavy_aug2 exists
                 img_array, mask_array = ImageProcessor.heavy_aug2(image=img_array, mask=mask_array)

        # --- Ensure 3 Channels for ResNet50 (CRITICAL FIX) ---
        if img_array.ndim == 2: # H, W (grayscale)
            # print(f"DEBUG: Converting 2D image ({img_array.shape}) to 3-channel.")
            img_array = np.stack([img_array]*3, axis=0) # Convert to C, H, W (3, H, W)
        elif img_array.ndim == 3 and img_array.shape[0] == 1: # 1, H, W (grayscale with channel dim)
            # print(f"DEBUG: Converting 1-channel 3D image ({img_array.shape}) to 3-channel.")
            img_array = np.concatenate([img_array]*3, axis=0) # Convert to 3, H, W
        elif img_array.ndim == 3 and img_array.shape[0] == 3: # Already C,H,W (3,H,W)
            # print(f"DEBUG: Image already 3-channel ({img_array.shape}).")
            pass
        else:
            # This case might occur if simple_aug/heavy_aug returns H,W,C and it's grayscale
            if img_array.ndim == 3 and img_array.shape[-1] == 1: # H,W,1
                # print(f"DEBUG: Converting H,W,1 image ({img_array.shape}) to 3-channel C,H,W.")
                img_array = img_array.squeeze(-1) # to H,W
                img_array = np.stack([img_array]*3, axis=0) # to 3,H,W
            elif img_array.ndim == 3 and img_array.shape[-1] == 3: # H,W,3
                # print(f"DEBUG: Converting H,W,3 image ({img_array.shape}) to 3-channel C,H,W.")
                img_array = np.moveaxis(img_array, -1, 0) # H,W,C -> C,H,W
            else:
                raise ValueError(f"Unsupported image shape before tensor conversion: {img_array.shape}")
        # --- End Ensure 3 Channels ---
            
        img_tensor = torch.from_numpy(img_array.astype(np.float32))
        mask_tensor = torch.from_numpy(mask_array.astype(np.int64)) 

        # print(f"DEBUG: Final img_tensor shape: {img_tensor.shape}") # Add this to check
        return img_tensor, mask_tensor, os.path.basename(img_path)


def prepare_dataset(args, aug_counter=False, data_dir_override=None, raw_data_dir_override=None):
    data_dir_to_use = data_dir_override if data_dir_override is not None else args.data_dir
    raw_data_dir_to_use = raw_data_dir_override if raw_data_dir_override is not None else args.raw_data_dir
    
    src_modality = 'ct' if not args.rev else 'mr'
    trg_modality = 'mr' if not args.rev else 'ct'

    train_s_dataset = DataGenerator(
        data_dir=data_dir_to_use, raw_data_dir=raw_data_dir_to_use,
        modality=src_modality, domain='s', fold=args.fold, split=args.split, bs=args.bs,
        augmentation=args.aug_s, num_class=args.num_classes, crop_size=args.crop,
        normalization=args.normalization, aug_mode=args.aug_mode, percent=args.percent, args_config=args
    )
    train_t_dataset = DataGenerator(
        data_dir=data_dir_to_use, raw_data_dir=raw_data_dir_to_use,
        modality=trg_modality, domain='t', fold=args.fold, split=args.split, bs=args.bs,
        augmentation=args.aug_t, num_class=args.num_classes, crop_size=args.crop,
        normalization=args.normalization, aug_mode=args.aug_mode, percent=args.percent, args_config=args
    )
    test_s_dataset = DataGenerator(
        data_dir=data_dir_to_use, raw_data_dir=raw_data_dir_to_use,
        modality=src_modality, domain='test', fold=args.fold, split=args.split, bs=args.eval_bs,
        augmentation=False, num_class=args.num_classes, crop_size=args.crop,
        normalization=args.normalization, aug_mode=args.aug_mode, percent=args.percent, args_config=args
    )
    test_t_dataset = DataGenerator(
        data_dir=data_dir_to_use, raw_data_dir=raw_data_dir_to_use,
        modality=trg_modality, domain='test', fold=args.fold, split=args.split, bs=args.eval_bs,
        augmentation=False, num_class=args.num_classes, crop_size=args.crop,
        normalization=args.normalization, aug_mode=args.aug_mode, percent=args.percent, args_config=args
    )
    
    num_workers = getattr(args, 'num_workers', 0)

    train_s_loader = DataLoader(train_s_dataset, batch_size=args.bs, shuffle=True, num_workers=num_workers, pin_memory=args.pin_memory, drop_last=True) if len(train_s_dataset) > 0 else None
    train_t_loader = DataLoader(train_t_dataset, batch_size=args.bs, shuffle=True, num_workers=num_workers, pin_memory=args.pin_memory, drop_last=True) if len(train_t_dataset) > 0 else None
    test_s_loader = DataLoader(test_s_dataset, batch_size=args.eval_bs, shuffle=False, num_workers=num_workers, pin_memory=args.pin_memory) if len(test_s_dataset) > 0 else None
    test_t_loader = DataLoader(test_t_dataset, batch_size=args.eval_bs, shuffle=False, num_workers=num_workers, pin_memory=args.pin_memory) if len(test_t_dataset) > 0 else None

    return {'train_s': train_s_loader, 'train_t': train_t_loader, 'test_s': test_s_loader, 'test_t': test_t_loader}