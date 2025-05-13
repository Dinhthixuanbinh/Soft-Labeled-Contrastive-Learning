# %%writefile /kaggle/working/Soft-Labeled-Contrastive-Learning/evaluator.py
import torch
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader 

# Assuming these are in the correct relative paths or sys.path is set
try:
    import config # Assuming config.py is at project root
    # Assuming these data generators are in dataset/
    from dataset.data_generator_mscmrseg import DataGenerator as MSCMRSEG_DataGenerator
    from dataset.data_generator_mmwhs import DataGenerator as MMWHS_DataGenerator
    from dataset.data_generator_mmwhs_raw import DataGenerator as MMWHS_Raw_DataGenerator
    # Assuming metric.py is at project root or in utils/
    from metric import dc, hd95, asd 
    from utils.utils_ import keep_largest_connected_components, save_nii 
except ImportError as e:
    print(f"Import Error in evaluator.py: {e}")
    import sys
    project_root = Path(os.getcwd()) 
    if "Soft-Labeled-Contrastive-Learning" not in str(project_root):
        # Try to find the project root assuming typical Kaggle structure
        if (Path.cwd() / "Soft-Labeled-Contrastive-Learning").exists():
             project_root = Path.cwd() / "Soft-Labeled-Contrastive-Learning"
        elif (Path.cwd().parent / "Soft-Labeled-Contrastive-Learning").exists():
             project_root = Path.cwd().parent / "Soft-Labeled-Contrastive-Learning"

    if str(project_root) not in sys.path and project_root.exists():
         sys.path.insert(0, str(project_root))
         print(f"Attempted to add {project_root} to sys.path in evaluator.py")
    
    # Retry imports - this is mostly for standalone execution of this file
    import config
    from dataset.data_generator_mscmrseg import DataGenerator as MSCMRSEG_DataGenerator
    from dataset.data_generator_mmwhs import DataGenerator as MMWHS_DataGenerator
    from dataset.data_generator_mmwhs_raw import DataGenerator as MMWHS_Raw_DataGenerator
    from metric import dc, hd95, asd
    from utils.utils_ import keep_largest_connected_components, save_nii


class Evaluator:
    def __init__(self, data_dir, raw_data_dir=None, raw=False, normalization='minmax', dataset='mscmrseg', args_config=None): # Added args_config
        self._data_dir = data_dir
        self._raw_data_dir = raw_data_dir if raw_data_dir is not None else data_dir
        self._raw = raw
        self._normalization = normalization
        self._dataset = dataset
        self.args_config = args_config # Store the arguments from the Trainer

        if self.args_config is None:
            print("WARNING: Evaluator initialized without args_config. Using fallback defaults for some parameters.")
            from argparse import Namespace
            self.args_config = Namespace(
                num_classes=getattr(config, 'NUM_CLASSES', 4), 
                crop=getattr(config, 'INPUT_SIZE', 224),
                fold=0, split=0, percent=99.0, spacing=1.0, rev=False,
                num_workers=0, pin_memory=False, eval_bs=16, # eval_bs for internal loader
                data_dir=data_dir, raw_data_dir=self._raw_data_dir, raw=raw, normalization=normalization
            )

        self._num_classes = getattr(self.args_config, 'num_classes', config.NUM_CLASSES)
        self._crop_size = getattr(self.args_config, 'crop', config.INPUT_SIZE)
        
        print(f"Evaluator initialized for dataset: {self._dataset}, raw: {self._raw}, norm: {self._normalization}")
        if self.args_config:
             print(f"Evaluator using args_config - fold: {getattr(self.args_config, 'fold', 'N/A')}, split: {getattr(self.args_config, 'split', 'N/A')}, percent: {getattr(self.args_config, 'percent', 'N/A')}")

    # MODIFIED: Added 'modality_name' to signature and uses self.args_config for other params
    def evaluate_single_dataset(self, seg_model, modality_name, phase, bs, toprint, klc, ifhd, ifasd, fold_num, split,
                                weight_dir=None, pred_index=0, 
                                save_prediction=False, log_dir=None):
        
        seg_model.eval()
        device = next(seg_model.parameters()).device

        # Use args_config stored during __init__ for consistent parameters
        # These are the full arguments from the Trainer
        current_args = self.args_config 
        
        # Parameters for this specific evaluation call (fold_num, split might be overridden)
        current_fold = fold_num if fold_num is not None else current_args.fold
        current_split = split if split is not None else current_args.split
        
        # Get these from self.args_config as they define how data was trained/should be processed
        # and how the DataGenerator for evaluation should be configured
        current_percent = current_args.percent
        current_spacing = current_args.spacing # Used by hd95, asd
        current_crop_size = current_args.crop
        current_normalization = current_args.normalization
        current_raw = current_args.raw
        # rev flag is important for DataGenerator to pick the right MMWHS subfolders
        current_rev = getattr(current_args, 'rev', False) 

        eval_dataset = None
        # Base path for data loading should come from the evaluator's init or current_args
        data_dir_for_eval_gen = self._data_dir # Path that might point to scratch
        raw_data_dir_for_eval_gen = self._raw_data_dir # Original raw data path

        if self._dataset == 'mscmrseg':
            eval_dataset = MSCMRSEG_DataGenerator(
                phase=phase, modality=modality_name, fold=current_fold, 
                data_dir=data_dir_for_eval_gen, # This is the main data path for mscmrseg pngs
                crop_size=current_crop_size, normalization=current_normalization, 
                args_config=current_args, # Pass the full args_config
                augmentation=False, bs=bs # bs here is eval_bs from Trainer
            )
        elif self._dataset == 'mmwhs':
            if current_raw:
                eval_dataset = MMWHS_Raw_DataGenerator(
                    data_dir=data_dir_for_eval_gen, # Not used by MMWHS_Raw_DataGenerator directly
                    raw_data_dir=raw_data_dir_for_eval_gen, # This is used for MMWHS raw .nii files
                    modality=modality_name, domain=phase, 
                    fold=current_fold, split=current_split, bs=bs, num_class=self._num_classes,
                    crop_size=current_crop_size, normalization=current_normalization,
                    percent=current_percent, augmentation=False, args_config=current_args
                )
            else: # MMWHS non-raw (preprocessed)
                eval_dataset = MMWHS_DataGenerator( 
                    data_dir=data_dir_for_eval_gen, modality=modality_name, domain=phase,
                    fold=current_fold, split=current_split, bs=bs, num_class=self._num_classes,
                    crop_size=current_crop_size, normalization=current_normalization,
                    percent=current_percent, augmentation=False, args_config=current_args
                )
        else:
            raise NotImplementedError(f"Dataset {self._dataset} not supported by Evaluator")

        if len(eval_dataset) == 0:
            print(f"Warning: Evaluation dataset for {modality_name} {phase} (fold {current_fold}, split {current_split}) is empty. Check paths and patient lists in DataGenerator. Skipping evaluation.")
            num_fg_classes = self._num_classes -1 if self._num_classes > 0 else 0
            return {'dc': [0.0]*num_fg_classes*2, 'hd': [np.inf]*num_fg_classes*2, 'asd': [np.inf]*num_fg_classes*2}

        eval_loader = DataLoader(eval_dataset, batch_size=bs, shuffle=False, 
                                 num_workers=getattr(current_args, 'num_workers',0), 
                                 pin_memory=getattr(current_args, 'pin_memory', False))

        dsc_list, hdn_list, asd_list = [], [], []
        pred_dir = None
        if save_prediction and log_dir:
            pred_dir = Path(log_dir) / f"eval_preds_{modality_name}_{phase}_f{current_fold}"
            pred_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for batch_idx, batch_data in tqdm(enumerate(eval_loader), total=len(eval_loader), desc=f"Evaluating {modality_name} {phase} F{current_fold} S{current_split}"):
                img, lab, names = batch_data
                img = img.to(device)
                
                outputs = seg_model(img)
                pred_all = outputs[pred_index] if isinstance(outputs, (tuple, list)) and len(outputs) > pred_index and outputs[pred_index] is not None else outputs
                
                pred_argmax = torch.argmax(pred_all, dim=1).cpu().numpy().astype(np.uint8)
                lab_np = lab.numpy().astype(np.uint8)

                for i in range(pred_argmax.shape[0]):
                    pred_slice = pred_argmax[i]
                    lab_slice = lab_np[i]
                    
                    if klc:
                        pred_slice = keep_largest_connected_components(pred_slice, num_total_classes=self._num_classes)

                    current_dc = dc(pred_slice, lab_slice) # dc takes 2 args
                    
                    if len(current_dc) == self._num_classes: 
                        dsc_list.append(current_dc[1:]) 
                    elif len(current_dc) == self._num_classes - 1:
                        dsc_list.append(current_dc)
                    else:
                        print(f"Warning: dc function returned {len(current_dc)} scores. Expected {self._num_classes} or {self._num_classes-1}.")
                        # Fallback: append as is or handle error
                        dsc_list.append(current_dc)


                    if ifhd:
                        current_hd = hd95(pred_slice, lab_slice, self._num_classes, voxelspacing=current_spacing, connectivity=1)
                        if len(current_hd) == self._num_classes: hdn_list.append(current_hd[1:])
                        else: hdn_list.append(current_hd)
                    if ifasd:
                        current_asd = asd(pred_slice, lab_slice, self._num_classes, voxelspacing=current_spacing)
                        if len(current_asd) == self._num_classes: asd_list.append(current_asd[1:])
                        else: asd_list.append(current_asd)
                    
                    if save_prediction and pred_dir and names and i < len(names):
                        try:
                            # np.save(pred_dir / f"{Path(names[i]).stem}_pred.npy", pred_slice)
                            pass 
                        except Exception as e_save:
                            print(f"Warning: Could not save prediction for {names[i]}: {e_save}")
        
        num_fg_classes = self._num_classes - 1 if self._num_classes > 0 else 0
        if num_fg_classes <= 0: # Handle case where num_classes might be 0 or 1
             print("Warning: Number of foreground classes is <= 0. Metrics will be empty or default.")
             return {'dc': [], 'hd': [], 'asd': []}


        dsc_arr = np.array(dsc_list) 
        mean_dsc = np.nanmean(dsc_arr, axis=0) if dsc_arr.size > 0 else np.zeros(num_fg_classes)
        std_dsc = np.nanstd(dsc_arr, axis=0) if dsc_arr.size > 0 else np.zeros(num_fg_classes)
        
        results = {
            'dc': [val for pair in zip(mean_dsc, std_dsc) for val in pair] if mean_dsc.size > 0 else [0.0]*num_fg_classes*2,
            'hd': [np.inf] * num_fg_classes * 2, 
            'asd': [np.inf] * num_fg_classes * 2
        }

        if ifhd and hdn_list:
            hdn_arr = np.array(hdn_list)
            mean_hdn = np.nanmean(hdn_arr, axis=0) if hdn_arr.size > 0 else np.full(num_fg_classes, np.inf)
            std_hdn = np.nanstd(hdn_arr, axis=0) if hdn_arr.size > 0 else np.zeros(num_fg_classes)
            results['hd'] = [val for pair in zip(mean_hdn, std_hdn) for val in pair] if mean_hdn.size > 0 and not np.all(np.isinf(mean_hdn)) else [np.inf]*num_fg_classes*2
        if ifasd and asd_list:
            asd_arr = np.array(asd_list)
            mean_asd = np.nanmean(asd_arr, axis=0) if asd_arr.size > 0 else np.full(num_fg_classes, np.inf)
            std_asd = np.nanstd(asd_arr, axis=0) if asd_arr.size > 0 else np.zeros(num_fg_classes)
            results['asd'] = [val for pair in zip(mean_asd, std_asd) for val in pair] if mean_asd.size > 0 and not np.all(np.isinf(mean_asd)) else [np.inf]*num_fg_classes*2

        if toprint:
            print(f"Results for {modality_name} {phase} Fold {current_fold} Split {current_split}:")
            avg_dice_overall = np.nanmean(mean_dsc) if mean_dsc.size > 0 else 0.0
            # Ensure loops don't go out of bounds if mean_dsc, mean_hdn, mean_asd are shorter
            for c_idx in range(len(mean_dsc)): 
                print(f"  Class {c_idx+1} DC: {mean_dsc[c_idx]:.3f} +/- {std_dsc[c_idx]:.3f}")
                if ifhd and hdn_list and c_idx < len(mean_hdn): print(f"  Class {c_idx+1} HD95: {mean_hdn[c_idx]:.3f} +/- {std_hdn[c_idx]:.3f}")
                if ifasd and asd_list and c_idx < len(mean_asd): print(f"  Class {c_idx+1} ASD: {mean_asd[c_idx]:.3f} +/- {std_asd[c_idx]:.3f}")
            print(f"  Average FG Dice: {avg_dice_overall:.3f}")
            if ifhd and hdn_list: print(f"  Average FG HD95: {np.nanmean(mean_hdn) if hdn_arr.size > 0 and not np.all(np.isinf(mean_hdn)) else np.inf:.3f}")
            if ifasd and asd_list: print(f"  Average FG ASD: {np.nanmean(mean_asd) if asd_arr.size > 0 and not np.all(np.isinf(mean_asd)) else np.inf:.3f}")
            
        return results