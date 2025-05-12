# %%writefile /kaggle/working/Soft-Labeled-Contrastive-Learning/trainer/Trainer_baseline.py
from datetime import datetime
import os
import numpy as np
from tqdm import tqdm
import argparse

import torch

# Assuming your dataset loaders are correctly structured relative to this file
from dataset.data_generator_mscmrseg import prepare_dataset
from dataset.data_generator_mmwhs import prepare_dataset as prepare_dataset_mmwhs
from dataset.data_generator_mmwhs_raw import prepare_dataset as prepare_dataset_mmwhs_raw
from utils.lr_adjust import adjust_learning_rate, adjust_learning_rate_custom
from utils.utils_ import save_batch_data # Ensure this is used or needed
from utils import timer
import config
from utils.loss import loss_calc
from trainer.Trainer import Trainer # Import base Trainer


class Trainer_baseline(Trainer):
    def __init__(self):
        super().__init__()
        # Specific initializations for Trainer_baseline, if any, can go here
        # Note: self.args.train_with_s default is now handled in Trainer.py get_argparser
        # and then potentially overridden by the launching script (train_baseline.py)

    def add_additional_arguments(self):
        # Call super if Trainer.add_additional_arguments is not abstract and has base args
        # super().add_additional_arguments() # If Trainer also adds args via this method
        
        # Arguments specific to Trainer_baseline
        # Ensure BooleanOptionalAction is used if Python 3.9+ for flags like train_with_s
        if not hasattr(self.parser._option_string_actions, '-train_with_s'): # Avoid re-adding
            self.parser.add_argument("-train_with_s", default=True, action=getattr(argparse, 'BooleanOptionalAction', 'store_true'))
        if not hasattr(self.parser._option_string_actions, '-train_with_t'):
            self.parser.add_argument("-train_with_t", action='store_true') # Default False
        
        self.parser.add_argument("-eval_bs", type=int, default=config.EVAL_BS,
                                 help="Number of images sent to the network in a batch during evaluation.")
        self.parser.add_argument('-toggle_klc',
                                 help='Whether to apply keep_largest_component in evaluation during training.',
                                 action='store_false', default=True) # Default to True (klc active)
        self.parser.add_argument('-hd95', action='store_true')
        self.parser.add_argument('-multilvl', help='if apply multilevel network', action='store_true')
        self.parser.add_argument('-estop', help='if apply early stop', action='store_true')
        self.parser.add_argument('-stop_epoch', type=int, default=200,
                                 help='The number of epochs as the tolerance to stop the training.')

    @timer.timeit
    def get_arguments_apdx(self):
        # This will be called by _initialize_training_resources in Trainer
        super().get_basic_arguments_apdx(name='Base') # Uses self.args already finalized
        # self.apdx += f".bs{self.args.bs}" # Already in get_basic_arguments_apdx
        # self.apdx += '.trainW' # Already in get_basic_arguments_apdx
        # if self.args.train_with_s: self.apdx += 's' # Already in get_basic_arguments_apdx
        # if self.args.train_with_t: self.apdx += 't' # Already in get_basic_arguments_apdx
        # Normalization part also in get_basic_arguments_apdx
        # No need to add more here unless specific to baseline AND not covered by base
        print(f'Final apdx for Trainer_baseline: {self.apdx}')


    @timer.timeit
    def prepare_dataloader(self):
        # Determine data_dir for scratch transfer correctly
        data_dir_to_use = self.args.data_dir
        raw_data_dir_to_use = self.args.raw_data_dir

        if self.args.scratch:
            self.scratch = tranfer_data_2_scratch(data_dir_to_use, self.args.scratch)
            self.scratch_raw = tranfer_data_2_scratch(raw_data_dir_to_use, self.args.scratch)
            # Update args to use scratch paths for DataGenerators
            # This mutable change to self.args is a bit risky but often done.
            # Or, pass self.scratch directly to prepare_dataset functions.
            # For now, let's assume prepare_dataset functions can handle self.args.scratch correctly if it's set.
            # Alternatively, make data_dir an attribute of the data generator args.
        else:
            self.scratch = data_dir_to_use
            self.scratch_raw = raw_data_dir_to_use


        if self.dataset == 'mscmrseg':
            _, _, self.content_loader, self.style_loader = prepare_dataset(self.args, data_dir=self.scratch, raw_data_dir=self.scratch_raw)
        elif self.dataset == 'mmwhs':
            print('Preparing MMWHS dataloader...')
            if self.args.raw:
                print(f"Using raw data loader for MMWHS from: {self.scratch}")
                _, _, self.content_loader, self.style_loader = prepare_dataset_mmwhs_raw(self.args, data_dir=self.scratch, raw_data_dir=self.scratch_raw)
            else:
                _, _, self.content_loader, self.style_loader = prepare_dataset_mmwhs(self.args, data_dir=self.scratch, raw_data_dir=self.scratch_raw)
        else:
            raise NotImplementedError


    @timer.timeit
    def prepare_model(self):
        # Model loading logic using self.args.backbone, self.args.pretrained etc.
        if self.args.backbone == 'unet':
            from model.unet_model import UNet
            self.segmentor = UNet(n_channels=3, n_classes=self.args.num_classes)
        elif self.args.backbone == 'drunet':
            from model.DRUNet import Segmentation_model as DR_UNet
            self.segmentor = DR_UNet(filters=self.args.filters, n_block=self.args.nb, bottleneck_depth=self.args.bd,
                                     n_class=self.args.num_classes, multilvl=self.args.multilvl, args=self.args)
        elif 'resnet' in self.args.backbone or any(enc in self.args.backbone for enc in ['efficientnet', 'mobilenet', 'densenet', 'ception', 'se_resnet', 'skresnext']):
            from model.segmentation_models import segmentation_models # Your wrapper
            print(f"Instantiating segmentation_models with backbone: {self.args.backbone}, pretrained: {self.args.pretrained}")
            self.segmentor = segmentation_models(
                name=self.args.backbone,
                pretrained=self.args.pretrained, # This will pass True to smp.Unet for ImageNet weights
                decoder_channels=(512, 256, 128, 64, 32), # Example
                in_channel=3,
                classes=self.args.num_classes,
                multilvl=self.args.multilvl,
                args=self.args # For phead etc.
            )
        else: # Fallback or error for other backbones like deeplabv2
            raise NotImplementedError(f"Backbone {self.args.backbone} not fully configured in this simplified prepare_model.")

        if self.args.restore_from:
            print(f"Restoring model from: {self.args.restore_from}")
            checkpoint = torch.load(self.args.restore_from, map_location=self.device)
            # If loading a full model checkpoint (not just encoder)
            if not self.args.pretrained: # Only try to load full state if not using the --pretrained flag (which implies fresh ImageNet encoder)
                try:
                    # Check for common state_dict keys
                    if 'model_state_dict' in checkpoint:
                        self.segmentor.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        print('Loaded model_state_dict from checkpoint.')
                    elif 'segmentor_state_dict' in checkpoint: # Another common key
                        self.segmentor.load_state_dict(checkpoint['segmentor_state_dict'], strict=False)
                        print('Loaded segmentor_state_dict from checkpoint.')
                    else: # Try loading the whole checkpoint object
                        self.segmentor.load_state_dict(checkpoint, strict=False)
                        print('Loaded entire checkpoint object as state_dict.')
                except Exception as e:
                    print(f"Could not load state_dict from restore_from checkpoint: {e}. Model might be partially loaded or random.")
            else:
                 print("Skipping restore_from for segmentor state_dict because args.pretrained is True (implies fresh ImageNet encoder + random decoder).")


            if 'epoch' in checkpoint and not self.args.pretrained: # Only load epoch if not starting fresh with ImageNet
                self.start_epoch = checkpoint['epoch'] + 1 # Start from next epoch
                print(f"Restored epoch: {self.start_epoch -1 }")


        self.segmentor.train()
        self.segmentor.to(self.device)
        print(f"Model {self.args.backbone} prepared on device: {self.device}")


    @timer.timeit
    def prepare_optimizers(self):
        # Simplified optimizer creation
        params = self.segmentor.parameters()
        if self.args.optim == 'sgd':
            self.opt = torch.optim.SGD(params, lr=self.args.lr, momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optim == 'adam':
            self.opt = torch.optim.Adam(params, lr=self.args.lr, betas=(0.9, 0.99), weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {self.args.optim} not implemented.")

        if self.args.restore_from and not self.args.pretrained : # Only load optimizer if also restoring model state
            checkpoint = torch.load(self.args.restore_from, map_location=self.device)
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
                    print(f"Optimizer state loaded from {os.path.basename(self.args.restore_from)}")
                except Exception as e:
                    print(f'Warning: Could not load optimizer state: {e}')
        self.opt.zero_grad()
        print(f'{self.args.optim.upper()} optimizer created.')

    # prepare_checkpoints, adjust_lr, eval, stop_training, train_epoch, train
    # remain mostly the same as your provided code, just ensure they use self.args

    @timer.timeit
    def train(self):
        self._initialize_training_resources() # This now sets up apdx, dataloaders, model, etc.

        for epoch in tqdm(range(self.start_epoch, self.args.epochs), desc="Training Baseline"):
            self.adjust_lr(epoch=epoch)
            epoch_start = datetime.now()

            train_results = self.train_epoch(epoch) # train_epoch uses self.content_loader / self.style_loader

            msg = f'Epoch = {epoch + 1:6d}/{self.args.epochs:6d}'
            if self.args.train_with_s and 'seg_s' in train_results:
                msg += f', loss_seg_s = {train_results["seg_s"]:.4f}'
            if self.args.train_with_t and 'seg_t' in train_results:
                msg += f', loss_seg_t = {train_results["seg_t"]:.4f}'
            
            valid_dice = 0.0 # Placeholder if eval is skipped
            if (epoch + 1) % 10 == 0 or epoch == self.args.epochs -1 : # Evaluate every 10 epochs and at the end
                results_eval = self.eval(modality='target' if self.args.train_with_t else 'source', phase='valid', toprint=False)
                # Assuming 'dc' is [myo_dc_mean, myo_dc_std, lv_dc_mean, lv_dc_std, rv_dc_mean, rv_dc_std]
                valid_dice_scores = [results_eval['dc'][k] for k in [0, 2, 4]] # myo, lv, rv means
                valid_dice = np.nanmean(valid_dice_scores) if valid_dice_scores else 0.0
                msg += f', val_dice = {valid_dice:.4f}'
            
            print(msg)

            tobreak = self.stop_training(epoch, epoch_start, valid_dice)
            self.mcp_segmentor.step(monitor=valid_dice, model=self.segmentor, epoch=epoch + 1,
                                    optimizer=self.opt, tobreak=tobreak)

            if self.args.train_with_s and 'seg_s' in train_results:
                self.writer.add_scalar('Loss_Seg/Source', train_results['seg_s'], epoch + 1)
            if self.args.train_with_t and 'seg_t' in train_results:
                 self.writer.add_scalar('Loss_Seg/Target', train_results['seg_t'], epoch + 1)
            self.writer.add_scalar('LR/Segmentor', self.opt.param_groups[0]['lr'], epoch + 1)
            if (epoch + 1) % 10 == 0 or epoch == self.args.epochs -1:
                self.writer.add_scalar('Dice/Valid_Target_AVG', valid_dice, epoch + 1)


            if tobreak:
                print(f"Stopping training at epoch {epoch+1}")
                break

        self.writer.close()
        best_epoch = self.mcp_segmentor.epoch
        best_score = self.mcp_segmentor.best_result
        
        final_log_dir_name = '{}.e{}.Scr{:.4f}'.format(self.apdx, best_epoch, best_score)
        try:
            os.rename(self.log_dir, Path(self.log_dir).parent / final_log_dir_name)
            print(f"Renamed log directory to: {Path(self.log_dir).parent / final_log_dir_name}")
        except Exception as e:
            print(f"Error renaming log directory: {e}. Current log_dir: {self.log_dir}")


        print("Best model saved at epoch: {} with score: {:.4f}".format(best_epoch, best_score))
        print("Best model weights at: {}".format(self.mcp_segmentor.best_model_save_dir))

        if os.path.exists(self.mcp_segmentor.best_model_save_dir):
            print("Loading best model for final test evaluation...")
            try:
                checkpoint = torch.load(self.mcp_segmentor.best_model_save_dir)
                if 'model_state_dict' in checkpoint:
                    self.segmentor.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.segmentor.load_state_dict(checkpoint)
                print("Best model loaded successfully.")
            except Exception as e:
                print(f"Error loading best model for final eval: {e}")
        else:
            print(f"Best model checkpoint not found at {self.mcp_segmentor.best_model_save_dir}")

        if self.args.train_with_s: # If source was trained, test on source domain test set
            print("\n--- Final Test Evaluation (Source Domain) ---")
            self.eval(modality='source', phase='test', toprint=True, fold=self.args.fold) # Test on the same fold for source
        
        # Always test on target domain if UDA setup, or if target only training
        print("\n--- Final Test Evaluation (Target Domain) ---")
        self.eval(modality='target', phase='test', toprint=True, fold=self.args.fold)
        # If doing cross-validation (e.g. 2-fold for MMWHS), you might want to test on the other fold too
        if self.dataset == 'mmwhs': # Example for MMWHS 2-fold cross-validation
             print(f"\n--- Final Test Evaluation (Target Domain - Fold {1-self.args.fold}) ---")
             self.eval(modality='target', phase='test', toprint=True, fold=1-self.args.fold)
        return