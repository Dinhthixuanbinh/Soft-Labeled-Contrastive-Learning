# %%writefile /kaggle/working/Soft-Labeled-Contrastive-Learning/dataset/data_generator_mscmrseg.py
import os.path
import re
from pathlib import Path
import math

import imgaug # Make sure imgaug is installed
import numpy as np
from torch.utils import data
import cv2
from glob import glob
import imgaug.augmenters as iaa
# import elasticdeform # Only if you use it in heavy_aug

# Assuming config and utils_ are accessible.
# The train_baseline.py should handle adding project root to sys.path
import config
from utils.utils_ import tranfer_data_2_scratch # Keep if prepare_dataset here uses it
from torch.utils.data import DataLoader # Used in prepare_dataset


def to_categorical(mask, num_classes, channel='channel_first'):
    """
    Convert label into categorical format (one-hot encoded)
    """
    if channel != 'channel_first' and channel != 'channel_last':
        assert False, r"channel should be either 'channel_first' or 'channel_last'"
    assert num_classes > 1, "num_classes should be greater than 1"
    unique = np.unique(mask)
    if len(unique) > 0: 
        assert np.max(unique) < num_classes, "maximum value in the mask should be smaller than the num_classes"
    
    if mask.ndim > 2:
        if mask.shape[1] == 1 and mask.ndim == 4: mask = np.squeeze(mask, axis=1)
        elif mask.shape[-1] == 1 and mask.ndim == 4: mask = np.squeeze(mask, axis=-1)
    elif mask.ndim == 3:
        if mask.shape[0] == 1: mask = np.squeeze(mask, axis=0)
        elif mask.shape[-1] == 1: mask = np.squeeze(mask, axis=-1)

    eye = np.eye(num_classes, dtype='uint8')
    output = eye[mask.astype(np.int_)] 
    if channel == 'channel_first':
        if output.ndim == 3: output = np.moveaxis(output, -1, 0)
        elif output.ndim == 4: output = np.moveaxis(output, -1, 1)
    return output

class ImageProcessor:
    @staticmethod
    def aug(image, mask): # This is likely your 'heavy_aug' or a more comprehensive one
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.1, 0.05), "y": (-0.1, 0.1)},
                    rotate=(-10, 10),
                    shear=(-12, 12),
                    order=[0, 1], # Mask (0, Nearest), Image (1, Linear)
                    cval=(0, 255), #Tuple for image, int for mask if different processing needed
                    mode='constant'
                )),
                # Add other iaa augmenters if this is your 'heavy_aug'
            ],
            random_order=True
        )
        
        # imgaug expects (H,W,C) or (N,H,W,C) for images
        # and (H,W) or (N,H,W) for masks
        is_batched = image.ndim == 4
        
        if not is_batched:
            image_batch = image[np.newaxis, ...]
            mask_batch = mask[np.newaxis, ...] if mask is not None else None
        else:
            image_batch = image
            mask_batch = mask
        
        # Ensure mask is of a type iaa expects (e.g., int32 for segmentation maps)
        if mask_batch is not None:
            mask_iaa = iaa.SegmentationMapsOnImage(mask_batch.astype(np.int32), shape=image_batch.shape[1:3]) # Use H,W from image
            image_aug, mask_aug_iaa = seq(images=image_batch.astype(np.uint8), segmentation_maps=mask_iaa)
            mask_aug = mask_aug_iaa.get_arr()
        else:
            image_aug = seq(images=image_batch.astype(np.uint8))
            mask_aug = mask_batch # Remains None

        if not is_batched:
            image_aug = image_aug[0]
            if mask_aug is not None:
                mask_aug = mask_aug[0]
                
        return image_aug, mask_aug

    @staticmethod
    def simple_aug(image, mask, ang=(-15, 15), translate_x=(-0.1, 0.1), translate_y=(-0.1, 0.1), scale=(0.9, 1.1)):
        # Your existing simple_aug implementation
        # Ensure it correctly handles image and mask (possibly None)
        # Make sure cv2.warpAffine borderValue is appropriate (float for image, int for mask)
        # (Using the version from your provided code for consistency)
        if image.ndim == 2: 
            rows, cols = image.shape
            is_multichannel_input = False
        elif image.ndim == 3 and image.shape[0] == 1: 
            rows, cols = image.shape[1], image.shape[2]
            image = image.squeeze(0) 
            is_multichannel_input = True 
            if mask is not None and mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
        elif image.ndim == 3 and image.shape[-1] == 1: 
            rows, cols = image.shape[0], image.shape[1]
            image = image.squeeze(-1)
            is_multichannel_input = True
            if mask is not None and mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
        elif image.ndim == 3 and image.shape[0] == 3: # (3, H, W)
            rows, cols = image.shape[1], image.shape[2]
            image = np.moveaxis(image, 0, -1) # C,H,W -> H,W,C for cv2
            is_multichannel_input = True 
        else:
            raise ValueError(f"Unsupported image shape for simple_aug: {image.shape}")

        rand_ang = np.random.randint(ang[0], ang[1]) if ang[0] != ang[1] else ang[0]
        rand_tr_x = np.random.uniform(translate_x[0], translate_x[1]) if translate_x[0] != translate_x[1] else translate_x[0]
        rand_tr_y = np.random.uniform(translate_y[0], translate_y[1]) if translate_y[0] != translate_y[1] else translate_y[0]
        rand_scale = np.random.uniform(scale[0], scale[1]) if scale[0] != scale[1] else scale[0]

        M = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=rand_ang, scale=rand_scale)
        M[0, 2] += rand_tr_x * cols
        M[1, 2] += rand_tr_y * rows
        
        border_value_img = image.min() if image.size > 0 else 0
        aug_image = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float(border_value_img))
        if mask is not None:
            aug_mask = cv2.warpAffine(mask, M, (cols, rows), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            aug_mask = None

        if np.random.random() < 0.5:
            aug_image = cv2.flip(aug_image, 1)
            if aug_mask is not None:
                aug_mask = cv2.flip(aug_mask, 1)
        
        if is_multichannel_input and aug_image.ndim == 3 and image.ndim == 3 and image.shape[0]==3 : # If input was (3,H,W)
            aug_image = np.moveaxis(aug_image, -1, 0) # H,W,C -> C,H,W
        elif is_multichannel_input and aug_image.ndim == 2 and (image.ndim == 3 and (image.shape[0]==1 or image.shape[-1]==1)): # if input was (1,H,W) or (H,W,1)
             pass # Becomes H,W. DataLoader will add channel dim if needed.
        
        return aug_image, aug_mask

    @staticmethod
    def heavy_aug(image, mask, ang=(-15, 15), translate_x=(-0.1, 0.1), translate_y=(-0.1, 0.1), scale=(0.8, 1.2),
                  alpha=None, sigma=None, grid_scale=None, grid_distort=None, aug_mode='heavy', vmax=255):
        # Using the version from your latest provided code for this method
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        aug_list = [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            sometimes(iaa.geometric.Rot90(k=imgaug.ALL)), # k=imgaug.ALL means (0,1,2,3)
        ]
        # The aug_mode check in your original code here had a slight logic issue
        # It should use the aug_mode parameter passed to this function
        if '2' not in aug_mode: # Standard heavy_aug
            aug_list += [
                sometimes(iaa.SomeOf((0, 3),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 2.0)),
                        iaa.AverageBlur(k=(2, 6)),
                        iaa.MedianBlur(k=(3, 5)),
                    ]),
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.1 * vmax), per_channel=0.5
                    ),
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.01, 0.05), size_percent=(0.04, 0.1),
                            per_channel=0.2
                    ),])
                ])),
            ]
        else: # heavy_aug2
            aug_list += [
                sometimes(iaa.SomeOf((0, 3), # Max 3 of these
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 2.0)),
                        iaa.AverageBlur(k=(2, 6)),
                        iaa.MedianBlur(k=(3, 5)),
                    ]),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * vmax), per_channel=0.5),
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout((0.01, 0.05), size_percent=(0.04, 0.1),per_channel=0.2),
                    ]),
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0),n_segments=(20, 200))),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                    sometimes(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0, 0.7)),
                        iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                    ])),
                    iaa.Invert(0.05, per_channel=True), 
                    iaa.Add((-10, 10), per_channel=0.5),
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # This requires elasticdeform
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                ])),
            ]

        seq = iaa.Sequential(aug_list, random_order=True)

        # Ensure image is HWC for imgaug, mask is HW
        if image.ndim == 3 and image.shape[0] <= 3: # C, H, W
            image_hwc = np.moveaxis(image, 0, -1).astype(np.uint8)
        elif image.ndim == 2: # H, W
            image_hwc = image.astype(np.uint8)
        else:
            image_hwc = image.astype(np.uint8) # Assume H, W, C

        if mask.ndim == 3 and mask.shape[0] ==1 : mask = mask.squeeze(0) # Make HW if (1,H,W)

        mask_iaa = iaa.SegmentationMapsOnImage(mask.astype(np.int32), shape=image_hwc.shape[:2])
        
        image_aug_hwc, mask_aug_iaa = seq(images=image_hwc, segmentation_maps=mask_iaa)
        mask_aug = mask_aug_iaa.get_arr()

        # Convert image back to C, H, W if it was originally
        if image.ndim == 3 and image.shape[0] <= 3:
            image_aug_chw = np.moveaxis(image_aug_hwc, -1, 0)
        else: # Was H,W or H,W,C -> output as H,W (or H,W,C)
            image_aug_chw = image_aug_hwc 
        
        return image_aug_chw, mask_aug

    # --- ADD THIS METHOD ---
    @staticmethod
    def crop_resize(image, target_size=(224, 224), interpolation=cv2.INTER_AREA, pad_value=0):
        """
        Crops the center of an image and resizes it.
        If image is smaller than target_size, it will be padded.
        Assumes image is HxW or (C, HxW) or (HxWxC)
        """
        is_mask = (interpolation == cv2.INTER_NEAREST)
        current_pad_value = 0 if is_mask else pad_value

        # Handle C,H,W format by temporarily converting to H,W,C for cv2
        original_ndim = image.ndim
        if original_ndim == 3 and image.shape[0] <= 3: # C, H, W (e.g. 1,H,W or 3,H,W)
            img_for_cv = np.moveaxis(image, 0, -1) # To H,W,C
        else: # Assumes H,W or H,W,C
            img_for_cv = image.copy()

        h, w = img_for_cv.shape[:2]
        th, tw = target_size

        if h < th or w < tw:
            delta_h = max(0, th - h)
            delta_w = max(0, tw - w)
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            img_for_cv = cv2.copyMakeBorder(img_for_cv, top, bottom, left, right, cv2.BORDER_CONSTANT, value=float(current_pad_value)) # Ensure float for border value
            h, w = img_for_cv.shape[:2]

        x1 = max(0, int(round((w - tw) / 2.)))
        y1 = max(0, int(round((h - th) / 2.)))
        
        cropped_image_hwc = img_for_cv[y1:y1 + th, x1:x1 + tw]

        if cropped_image_hwc.shape[0] != th or cropped_image_hwc.shape[1] != tw: # Should not happen if padding is correct
            resized_image_hwc = cv2.resize(cropped_image_hwc, (tw, th), interpolation=interpolation)
        else:
            resized_image_hwc = cropped_image_hwc
        
        # Convert back to C,H,W if original was C,H,W
        if original_ndim == 3 and image.shape[0] <= 3:
            final_image = np.moveaxis(resized_image_hwc, -1, 0)
        else:
            final_image = resized_image_hwc
            
        return final_image
    # --- END ADDED METHOD ---

# ... (Rest of your DataGenerator class and prepare_dataset function for MS-CMRSeg) ...
# Ensure the DataGenerator class and prepare_dataset function here are the
# correct ones for the MS-CMRSeg dataset as per your project structure.
# The prepare_dataset in this file should be the one that Trainer_baseline imports
# when self.dataset == 'mscmrseg'.

# Minimal DataGenerator for MS-CMRSeg (from your provided structure)
class DataGenerator(data.Dataset):
    def __init__(self, phase="train", modality="bssfp", crop_size=224, n_samples=-1, augmentation=False, clahe=False,
                 data_dir='../data/mscmrseg/origin', pat_id=-1, slc_id=-1, bs=16, aug_mode='simple', aug_counter=False,
                 normalization='minmax', fold=0, domain='s', vert=False, args_config=None):
        self._modality = modality
        self._crop_size = crop_size
        self._phase = phase
        self._augmentation = augmentation
        self._aug_mode = aug_mode
        self._normalization = normalization
        self._ifclahe = clahe
        self.args_config = args_config # Store for use if needed
        
        if clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        
        pat_name = 'bSSFP' if modality == 'bssfp' else modality.upper()
        st_char = 'A' if (modality == 'bssfp' or modality == 't2') else 'B'
        
        # Determine patient IDs (This is simplified, use your config.py lists)
        # Example:
        if domain == 's': # Source, e.g., all training patients
            patient_ids_to_load = getattr(config, f"MSCMRSEG_{pat_name}_TRAIN", list(range(1, 36))) # Fallback
        elif domain == 't': # Target, could be a specific fold for UDA
            patient_ids_to_load = config.MSCMRSEG_TEST_FOLD1 if fold == 0 else config.MSCMRSEG_TEST_FOLD2
        elif domain == 'test':
            patient_ids_to_load = config.MSCMRSEG_TEST_FOLD1 if fold == 0 else config.MSCMRSEG_TEST_FOLD2
            if fold not in [0,1]: # If fold is not 0 or 1, use all test patients
                patient_ids_to_load = config.MSCMRSEG_TEST_FOLD1 + config.MSCMRSEG_TEST_FOLD2
        else:
            raise ValueError(f"Unknown domain '{domain}' for MS-CMRSeg DataGenerator")

        self._image_files = []
        self._mask_files = []
        search_folder_prefix = f"{phase}{st_char}" # e.g. trainA, testB

        for p_id in patient_ids_to_load:
            self._image_files.extend(sorted(glob(os.path.join(data_dir, search_folder_prefix, f'pat_{p_id}_{pat_name}_*.png'))))
            self._mask_files.extend(sorted(glob(os.path.join(data_dir, f'{search_folder_prefix}mask', f'pat_{p_id}_{pat_name}_*.png'))))

        if not self._image_files: print(f"Warning: No images found for MS-CMRSeg {modality} {phase} {domain} fold {fold}")
        self._len = len(self._image_files)
        self._n_samples = self._len if n_samples == -1 else n_samples
        self._names = [Path(file).stem for file in self._image_files]

    def __len__(self):
        return self._n_samples

    def get_images_masks(self, img_path, mask_path):
        img = cv2.imread(img_path) 
        if img is None: raise FileNotFoundError(f"Img not found: {img_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        mask_p = np.zeros_like(mask, dtype=np.uint8)
        mask_p[mask == 85] = 1  # LV
        mask_p[mask == 212] = 2 # RV
        mask_p[mask == 255] = 3 # MYO
        return img, mask_p

    def __getitem__(self, index):
        i = index % self._len
        img, mask = self.get_images_masks(self._image_files[i], self._mask_files[i])

        if self._ifclahe:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = self.clahe.apply(img_yuv[:,:,0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        aug_img, aug_mask = img, mask
        if self._augmentation:
            if self._aug_mode == 'simple':
                aug_img, aug_mask = ImageProcessor.simple_aug(image=img.astype(np.uint8), mask=mask.astype(np.uint8))
            elif self._aug_mode == 'heavy' or self._aug_mode == 'heavy2':
                aug_img, aug_mask = ImageProcessor.heavy_aug(image=img.astype(np.uint8), mask=mask.astype(np.uint8), aug_mode=self._aug_mode)
        
        if self._crop_size:
            aug_img = ImageProcessor.crop_resize(aug_img, target_size=(self._crop_size, self._crop_size), interpolation=cv2.INTER_LINEAR)
            aug_mask = ImageProcessor.crop_resize(aug_mask, target_size=(self._crop_size, self._crop_size), interpolation=cv2.INTER_NEAREST)
        
        aug_img = np.moveaxis(aug_img, -1, 0).astype(np.float32) # HWC to CHW
        
        if self._normalization == 'minmax': aug_img = aug_img / 255.0
        elif self._normalization == 'zscore':
            mean = aug_img.mean(axis=(1,2), keepdims=True)
            std = aug_img.std(axis=(1,2), keepdims=True)
            aug_img = (aug_img - mean) / (std + 1e-7)
        
        return torch.from_numpy(aug_img), torch.from_numpy(aug_mask.astype(np.int64)), self._names[i]

# This is what Trainer_baseline calls for mscmrseg
def prepare_dataset(args, aug_counter=False, data_dir_override=None, raw_data_dir_override=None):
    data_dir_to_use = data_dir_override if data_dir_override is not None else args.data_dir
    # MS-CMRSeg generator uses data_dir, not raw_data_dir in its current form
    
    src_modality = 'lge' if args.rev else 'bssfp'
    trg_modality = 'bssfp' if args.rev else 'lge'

    content_dataset = DataGenerator(
        phase="train", modality=src_modality, crop_size=args.crop, augmentation=args.aug_s,
        data_dir=data_dir_to_use, bs=args.bs, clahe=args.clahe, aug_mode=args.aug_mode,
        normalization=args.normalization, fold=args.fold, domain='s', args_config=args
    )
    style_dataset = DataGenerator(
        phase="train", modality=trg_modality, crop_size=args.crop, augmentation=args.aug_t,
        data_dir=data_dir_to_use, bs=args.bs, clahe=args.clahe, aug_mode=args.aug_mode,
        normalization=args.normalization, fold=args.fold, domain='t', args_config=args
    )
    test_s_dataset = DataGenerator(
        phase="test", modality=src_modality, crop_size=args.crop, augmentation=False,
        data_dir=data_dir_to_use, bs=args.eval_bs, clahe=args.clahe, aug_mode=args.aug_mode,
        normalization=args.normalization, fold=args.fold, domain='test', args_config=args
    )
    test_t_dataset = DataGenerator(
        phase="test", modality=trg_modality, crop_size=args.crop, augmentation=False,
        data_dir=data_dir_to_use, bs=args.eval_bs, clahe=args.clahe, aug_mode=args.aug_mode,
        normalization=args.normalization, fold=args.fold, domain='test', args_config=args
    )

    num_workers = getattr(args, 'num_workers', 0)
    train_s_loader = DataLoader(content_dataset, batch_size=args.bs, shuffle=True, num_workers=num_workers, pin_memory=args.pin_memory, drop_last=True) if len(content_dataset) > 0 else None
    train_t_loader = DataLoader(style_dataset, batch_size=args.bs, shuffle=True, num_workers=num_workers, pin_memory=args.pin_memory, drop_last=True) if len(style_dataset) > 0 else None
    test_s_loader = DataLoader(test_s_dataset, batch_size=args.eval_bs, shuffle=False, num_workers=num_workers, pin_memory=args.pin_memory) if len(test_s_dataset) > 0 else None
    test_t_loader = DataLoader(test_t_dataset, batch_size=args.eval_bs, shuffle=False, num_workers=num_workers, pin_memory=args.pin_memory) if len(test_t_dataset) > 0 else None

    return {'train_s': train_s_loader, 'train_t': train_t_loader, 'test_s': test_s_loader, 'test_t': test_t_loader}