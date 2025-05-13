# %%writefile /kaggle/working/Soft-Labeled-Contrastive-Learning/trainer/Trainer_baseline.py
from datetime import datetime
import os
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

import torch

# Assuming your dataset loaders are correctly structured relative to this file
from dataset.data_generator_mscmrseg import prepare_dataset # MS-CMRSeg version
from dataset.data_generator_mmwhs import prepare_dataset as prepare_dataset_mmwhs # MMWHS non-raw
# Correctly alias the function from data_generator_mmwhs_raw
from dataset.data_generator_mmwhs_raw import prepare_dataset as prepare_dataset_mmwhs_raw_actual_func_name 

from utils.lr_adjust import adjust_learning_rate, adjust_learning_rate_custom
from utils.utils_ import save_batch_data, tranfer_data_2_scratch
from utils import timer
import config
from utils.loss import loss_calc
from trainer.Trainer import Trainer # Import base Trainer


class Trainer_baseline(Trainer):
    def __init__(self):
        super().__init__()
        # Args are initialized by super().__init__()

    def add_additional_arguments(self):
        parser = self.parser 
        arg_dest_names = [action.dest for action in parser._actions]

        if 'train_with_s' not in arg_dest_names:
            parser.add_argument("-train_with_s", default=True, action=getattr(argparse, 'BooleanOptionalAction', 'store_true'))
        if 'train_with_t' not in arg_dest_names:
            parser.add_argument("-train_with_t", action='store_false') 
        
        if 'eval_bs' not in arg_dest_names:
            parser.add_argument("-eval_bs", type=int, default=config.EVAL_BS)
        
        if 'apply_klc_eval' not in arg_dest_names: 
            parser.add_argument('--toggle_klc', action='store_false', dest='apply_klc_eval', help="Disable KLC in evaluation during training.")
            parser.set_defaults(apply_klc_eval=True)

        if 'hd95' not in arg_dest_names:
            parser.add_argument('-hd95', action='store_true', help="Calculate HD95 metric during test.")
        
        # -multilvl is defined in base Trainer.py, ensure default is appropriate or set in launch script
        if 'multilvl' not in arg_dest_names: 
             parser.add_argument('-multilvl', default=False, action=getattr(argparse, 'BooleanOptionalAction', 'store_true'), help="Model has multi-level outputs.")


        if 'estop' not in arg_dest_names:
            parser.add_argument('-estop', action='store_true', help="Enable early stopping.")
        if 'stop_epoch' not in arg_dest_names:
            parser.add_argument('-stop_epoch', type=int, default=200, help="Patience for early stopping.")

    @timer.timeit
    def get_arguments_apdx(self):
        super().get_basic_arguments_apdx(name='Base')
        self.apdx += '.train' 
        if getattr(self.args, 'train_with_s', False): self.apdx += 'S'
        if getattr(self.args, 'train_with_t', False): self.apdx += 'T'
        print(f'Final apdx for Trainer_baseline: {self.apdx}')

    @timer.timeit
    def prepare_dataloader(self):
        data_dir_for_loader = self.args.data_dir
        raw_data_dir_for_loader = self.args.raw_data_dir

        if getattr(self.args, 'scratch', False):
            self.scratch = tranfer_data_2_scratch(self.args.data_dir, self.args.scratch)
            self.scratch_raw = tranfer_data_2_scratch(self.args.raw_data_dir, self.args.scratch)
            data_dir_for_loader = self.scratch
            raw_data_dir_for_loader = self.scratch_raw
        else:
            self.scratch = self.args.data_dir
            self.scratch_raw = self.args.raw_data_dir
        
        loader_args = self.args

        if self.dataset == 'mscmrseg':
            datasets_dict = prepare_dataset( 
                loader_args, 
                data_dir_override=data_dir_for_loader, 
                raw_data_dir_override=raw_data_dir_for_loader 
            )
        elif self.dataset == 'mmwhs':
            print('Preparing MMWHS dataloader...')
            if self.args.raw:
                print(f"Using raw data loader for MMWHS.")
                datasets_dict = prepare_dataset_mmwhs_raw_actual_func_name( 
                    loader_args, 
                    data_dir_override=data_dir_for_loader, 
                    raw_data_dir_override=raw_data_dir_for_loader
                )
            else:
                datasets_dict = prepare_dataset_mmwhs( 
                    loader_args, 
                    data_dir_override=data_dir_for_loader, 
                    raw_data_dir_override=raw_data_dir_for_loader
                )
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not implemented.")

        self.content_loader = datasets_dict.get('train_s')
        self.style_loader = datasets_dict.get('train_t')
        self.test_loader_s = datasets_dict.get('test_s')
        self.test_loader_t = datasets_dict.get('test_t')
        
        if self.args.train_with_s and not self.content_loader:
            print(f"Warning: train_with_s is True, but content_loader (train_s) is None or empty for {self.dataset} fold {self.args.fold} split {self.args.split}")
        if self.args.train_with_t and not self.style_loader:
            print(f"Warning: train_with_t is True, but style_loader (train_t) is None or empty for {self.dataset} fold {self.args.fold} split {self.args.split}")

        print(f"Content loader (train_s): {'Loaded with ' + str(len(self.content_loader.dataset)) + ' samples' if self.content_loader and hasattr(self.content_loader, 'dataset') else 'None or empty'}")
        print(f"Style loader (train_t): {'Loaded with ' + str(len(self.style_loader.dataset)) + ' samples' if self.style_loader and hasattr(self.style_loader, 'dataset') else 'None or empty'}")

    @timer.timeit
    def prepare_model(self):
        if self.args.backbone == 'unet':
            from model.unet_model import UNet
            self.segmentor = UNet(n_channels=3, n_classes=self.args.num_classes)
        elif self.args.backbone == 'drunet':
            from model.DRUNet import Segmentation_model as DR_UNet
            self.segmentor = DR_UNet(filters=self.args.filters, n_block=self.args.nb, bottleneck_depth=self.args.bd,
                                     n_class=self.args.num_classes, multilvl=self.args.multilvl, args=self.args)
        elif 'resnet' in self.args.backbone or any(enc in self.args.backbone for enc in ['efficientnet', 'mobilenet', 'densenet', 'ception', 'se_resnet', 'skresnext']):
            from model.segmentation_models import segmentation_models # Your wrapper
            print(f"Instantiating segmentation_models with: name={self.args.backbone}, pretrained={self.args.pretrained}, multilvl={self.args.multilvl}")
            self.segmentor = segmentation_models(
                name=self.args.backbone,
                pretrained=self.args.pretrained, # Passed to smp.Unet for ImageNet weights
                decoder_channels=(256, 128, 64, 32, 16), 
                in_channel=3,
                classes=self.args.num_classes,
                multilvl=self.args.multilvl, 
                args=self.args
            )
        else:
            raise NotImplementedError(f"Backbone {self.args.backbone} not in Trainer_baseline prepare_model.")

        if self.args.restore_from:
            print(f"Restoring model from checkpoint: {self.args.restore_from}")
            checkpoint = torch.load(self.args.restore_from, map_location=self.device)
            load_success = False
            try:
                state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'segmentor_state_dict'
                if state_dict_key in checkpoint:
                    self.segmentor.load_state_dict(checkpoint[state_dict_key], strict=False)
                    load_success = True
                else: 
                    self.segmentor.load_state_dict(checkpoint, strict=False)
                    load_success = True
                print(f"Loaded state_dict from checkpoint ('{state_dict_key if state_dict_key in checkpoint else 'whole object'}').")
            except Exception as e_load:
                print(f"Failed to load state_dict from checkpoint: {e_load}")

            if load_success and 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
                print(f"Restored training to start from epoch: {self.start_epoch}")
        
        self.segmentor.train()
        self.segmentor.to(self.device)
        print(f"Model {self.args.backbone} prepared. Trainable params: {sum(p.numel() for p in self.segmentor.parameters() if p.requires_grad):,}")

    @timer.timeit
    def prepare_optimizers(self):
        params = self.segmentor.parameters()
        if self.args.optim == 'sgd':
            self.opt = torch.optim.SGD(params, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optim == 'adam':
            self.opt = torch.optim.Adam(params, lr=self.args.lr, betas=(0.9, 0.99), weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {self.args.optim} not implemented.")

        if self.args.restore_from:
            checkpoint = torch.load(self.args.restore_from, map_location=self.device)
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Optimizer state loaded from checkpoint.")
                except Exception as e:
                    print(f'Warning: Could not load optimizer state: {e}')
        self.opt.zero_grad()
        print(f'{self.args.optim.upper()} optimizer created.')

    @timer.timeit
    def prepare_checkpoints(self, mode='max'): 
        from utils.callbacks import ModelCheckPointCallback, EarlyStopCallback
        weight_root_dir = Path('./weights/')
        weight_root_dir.mkdir(parents=True, exist_ok=True)
        
        weight_filename = f"{self.apdx}.pt"
        best_weight_filename = f"best_{self.apdx}.pt"
        
        self.mcp_segmentor = ModelCheckPointCallback(n_epochs=self.args.epochs, save_best=True,
                                                     mode=mode, best_model_dir=str(weight_root_dir / best_weight_filename),
                                                     save_last_model=True, model_name=str(weight_root_dir / weight_filename),
                                                     entire_model=False)
        self.earlystop = EarlyStopCallback(mode=mode, stop_criterion_len=self.args.stop_epoch if getattr(self.args, 'estop', False) else (self.args.epochs + 1))
        print('Model checkpointing setup complete.')

    def adjust_lr(self, epoch_num):
        if self.args.lr_decay_method == 'poly':
            adjust_learning_rate(optimizer=self.opt, epoch=epoch_num, lr=self.args.lr, 
                                 power=self.args.power, epochs=self.args.epochs)
        elif self.args.lr_decay_method == 'linear':
            adjust_learning_rate_custom(optimizer=self.opt, lr=self.args.lr, lr_decay=self.args.lr_decay, epoch=epoch_num)

    def train_epoch(self, epoch_num):
        print(f'Starting epoch: {epoch_num + 1}/{self.args.epochs}')
        results = {}
        loss_seg_list = []
        
        current_loader = None
        current_domain_name = "N/A"
        current_loss_key = "seg_loss" 

        if self.args.train_with_s:
            current_loader = self.content_loader
            current_domain_name = "Source"
            current_loss_key = "seg_s"
        elif self.args.train_with_t: 
            current_loader = self.style_loader 
            current_domain_name = "Target"
            current_loss_key = "seg_t"
        
        if not current_loader:
            print(f"Warning: No data loader active for epoch {epoch_num + 1}. Skipping training for this epoch.")
            results[current_loss_key] = 0.0
            return results

        for batch_data in tqdm(current_loader, desc=f"Epoch {epoch_num+1} ({current_domain_name})"):
            self.segmentor.train()
            self.opt.zero_grad()
            
            img_data, labels_data, _ = batch_data
            img_data = img_data.to(self.device, non_blocking=self.args.pin_memory)
            labels_data = labels_data.to(self.device, non_blocking=self.args.pin_memory)

            out = self.segmentor(img_data)
            pred = out[0] if isinstance(out, tuple) and len(out)>0 else out # Handle single tensor or tuple output

            loss_seg = loss_calc(pred, labels_data, self.device, jaccard=True)
            loss_seg_list.append(loss_seg.item())
            
            loss_seg.backward()
            self.opt.step()

        results[current_loss_key] = sum(loss_seg_list) / len(loss_seg_list) if loss_seg_list else 0.0
        return results

    @timer.timeit
    def train(self):
        self._initialize_training_resources() # CRITICAL: Call initialization here

        for epoch_idx in range(self.start_epoch, self.args.epochs):
            self.adjust_lr(epoch_idx)
            epoch_start_time = datetime.now()

            train_results = self.train_epoch(epoch_idx)

            msg = f'Epoch = {epoch_idx + 1:6d}/{self.args.epochs:6d}'
            loss_val_for_log = 0.0
            training_domain_for_log = "Unknown" 
            if self.args.train_with_s and 'seg_s' in train_results:
                msg += f', loss_seg_s = {train_results["seg_s"]:.4f}'
                loss_val_for_log = train_results["seg_s"]
                training_domain_for_log = "Source"
            elif self.args.train_with_t and 'seg_t' in train_results:
                msg += f', loss_seg_t = {train_results["seg_t"]:.4f}'
                loss_val_for_log = train_results["seg_t"]
                training_domain_for_log = "Target"
            
            valid_dice_avg = 0.0
            eval_frequency = getattr(self.args, 'eval_frequency', 10) 
            if (epoch_idx + 1) % eval_frequency == 0 or (epoch_idx + 1) == self.args.epochs:
                # For baseline, validate on the domain it's training on
                validation_modality = self.src_modality if self.args.train_with_s else self.trgt_modality
                if self.args.train_with_s and self.args.train_with_t: # If somehow training both (not typical for this baseline)
                    validation_modality = self.trgt_modality # Default to target
                
                eval_results = self.eval(modality=validation_modality, phase='valid', toprint=False, fold=self.args.fold)
                if 'dc' in eval_results and isinstance(eval_results['dc'], (list, np.ndarray)) and len(eval_results['dc']) >= (self.args.num_classes-1)*2 :
                     # Assuming dc is [m1,s1,m2,s2,m3,s3] for 3 FG classes
                    fg_dice_means = [eval_results['dc'][k*2] for k in range(self.args.num_classes-1)]
                    valid_dice_avg = np.nanmean(fg_dice_means) if fg_dice_means else 0.0
                msg += f', val_dice_avg ({validation_modality}) = {valid_dice_avg:.4f}'
            print(msg)
            
            tobreak = self.stop_training(epoch_idx, epoch_start_time, valid_dice_avg)
            if hasattr(self, 'mcp_segmentor') and self.mcp_segmentor: 
                self.mcp_segmentor.step(monitor=valid_dice_avg, model=self.segmentor, epoch=epoch_idx + 1,
                                        optimizer=self.opt, tobreak=tobreak)

            if self.writer: 
                 if training_domain_for_log != "Unknown" and f'seg_{training_domain_for_log.lower()}' in train_results:
                    self.writer.add_scalar(f'Loss_Seg/Train_{training_domain_for_log}', train_results[f'seg_{training_domain_for_log.lower()}'], epoch_idx + 1)
                 if self.opt and hasattr(self.opt, 'param_groups') and len(self.opt.param_groups) > 0: 
                    self.writer.add_scalar('LR/Segmentor', self.opt.param_groups[0]['lr'], epoch_idx + 1)
                 if (epoch_idx + 1) % eval_frequency == 0 or (epoch_idx + 1) == self.args.epochs:
                    self.writer.add_scalar('Dice/Valid_AVG', valid_dice_avg, epoch_idx + 1)

            if tobreak: break

        if self.writer: self.writer.close()
        if hasattr(self, 'mcp_segmentor') and self.mcp_segmentor.epoch > 0:
            best_epoch = self.mcp_segmentor.epoch
            best_score = self.mcp_segmentor.best_result if self.mcp_segmentor.best_result is not None else 0.0
            final_log_dir_name = f'{self.apdx}.e{best_epoch}.Scr{best_score:.4f}'
            try:
                new_log_dir_path = Path(self.log_dir).parent / final_log_dir_name
                if Path(self.log_dir).exists() and not new_log_dir_path.exists():
                     os.rename(self.log_dir, new_log_dir_path)
            except Exception as e: print(f"Error renaming log dir: {e}")
            print(f"Best model epoch {best_epoch}, score {best_score:.4f}, path: {self.mcp_segmentor.best_model_save_dir}")
            if os.path.exists(self.mcp_segmentor.best_model_save_dir):
                checkpoint = torch.load(self.mcp_segmentor.best_model_save_dir)
                load_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'segmentor_state_dict'
                if load_key in checkpoint: self.segmentor.load_state_dict(checkpoint[load_key])
                else: self.segmentor.load_state_dict(checkpoint)
                print("Loaded best model for final test evaluation.")
        else:
            print("No best model recorded. Evaluating last model state.")

        print("\n--- Final Test Evaluation ---")
        test_modality_src = self.src_modality
        test_modality_trg = self.trgt_modality
        
        if self.args.train_with_s:
            self.eval(modality=test_modality_src, phase='test', toprint=True, fold=self.args.fold)
        
        self.eval(modality=test_modality_trg, phase='test', toprint=True, fold=self.args.fold)
        
        if self.dataset == 'mmwhs' : 
             print(f"\n--- Final Test Evaluation ({test_modality_trg} - Fold {1-self.args.fold}) ---")
             self.eval(modality=test_modality_trg, phase='test', toprint=True, fold=1-self.args.fold)
        return

    def eval(self, modality='target', phase='valid', toprint=None, fold=None, **kwargs):
        current_fold = self.args.fold if fold is None else fold
        print(f"Evaluating on: Modality={modality}, Phase={phase}, Fold={current_fold}")
        
        if not hasattr(self, 'evaluator') or self.evaluator is None:
            print("ERROR: Evaluator not initialized in Trainer_baseline.eval")
            num_eval_classes = getattr(self.args, 'num_classes', 4)
            # Ensure results dict has enough values for later unpacking if 'dc' is used
            return {'dc': [0.0]*(num_eval_classes-1)*2, 'hd': [np.inf]*(num_eval_classes-1)*2, 'asd': [np.inf]*(num_eval_classes-1)*2} 
            
        # --- CORRECTED CALL TO EVALUATOR ---
        results = self.evaluator.evaluate_single_dataset(
            seg_model=self.segmentor,
            modality_name=modality, # Evaluator will use this to create its own dataloader
            phase=phase,
            bs=self.args.eval_bs, 
            toprint=True if toprint is None else toprint,
            klc=getattr(self.args, 'apply_klc_eval', True), 
            ifhd=getattr(self.args, 'hd95', False) if phase == 'test' else False,
            ifasd=getattr(self.args, 'hd95', False) if phase == 'test' else False, # Assuming hd95 controls asd too
            fold_num=current_fold, 
            split=self.args.split
            # The Evaluator's evaluate_single_dataset should use its stored self.args_config
            # for other parameters like percent, spacing, crop_size.
        )
        # --- END CORRECTION ---
        return results