# %%writefile /kaggle/working/Soft-Labeled-Contrastive-Learning/trainer/Trainer.py
import torch.backends.cudnn as cudnn
import torch
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import os
import argparse
import random
import numpy as np
from pathlib import Path

from utils.utils_ import get_device, check_bit_generator, print_device_info, get_summarywriter
import config
from evaluator import Evaluator


class Trainer(ABC):
    def __init__(self):
        self.start_time = datetime.now()
        self.max_epoch_time = 0
        self.start_epoch = 0
        self.max_duration = 24 * 3600 - 5 * 60 
        self.device = get_device()
        self.cores = os.cpu_count()
        self.get_argparser() 
        self.apdx = "initial_apdx" 
        self.writer = None
        self.log_dir = None
        self.scratch = None
        self.scratch_raw = None
        self.content_loader = None
        self.style_loader = None
        self.segmentor = None
        self.opt = None
        self.mcp_segmentor = None
        self.earlystop = None
        self.evaluator = None

    def get_argparser(self):
        self.parser = argparse.ArgumentParser()
        # General arguments
        self.parser.add_argument('-apdx', type=str, default='', help="Appendix for output filenames and log directories.")
        self.parser.add_argument('-evalT', action='store_true', help="Evaluate on test set during training epochs.")
        self.parser.add_argument('-val_num', type=int, default=0, help="Validation run number identifier.")
        self.parser.add_argument('-seed', type=int, default=config.RANDOM_SEED, help="Random seed.")
        self.parser.add_argument('-epochs', type=int, default=config.EPOCHS, help="Number of training epochs.")
        
        # Data arguments
        self.parser.add_argument('-data_dir', type=str, default=config.DATA_DIRECTORY, help="Main data directory.")
        self.parser.add_argument("-raw_data_dir", type=str, default=config.RAW_DATA_DIRECTORY, help="Directory for raw data if different.")
        self.parser.add_argument('-fold', type=int, default=0, help="Cross-validation fold number.")
        self.parser.add_argument('-split', type=int, default=0, help="Data split number for MMWHS.")
        self.parser.add_argument('-bs', type=int, default=config.BATCH_SIZE, help="Batch size.")
        self.parser.add_argument('-num_workers', type=int, default=-1, help="Num workers for DataLoader (-1 for all cores up to bs).")
        self.parser.add_argument("-raw", action='store_true', default=True, help="Use raw data loader (e.g., for .nii MMWHS).") # Default to True for MMWHS
        self.parser.add_argument("-percent", type=float, default=99.0, help="Percentile for min/max normalization.") # Default to 99
        self.parser.add_argument('-crop', type=int, default=config.INPUT_SIZE, help="Crop size for input images.")
        self.parser.add_argument('-normalization', type=str, default='minmax', choices=['minmax', 'zscore', 'none'], help="Normalization type.")
        self.parser.add_argument('-aug_s', action='store_true', help="Apply augmentation to source domain.")
        self.parser.add_argument('-aug_t', action='store_true', help="Apply augmentation to target domain.")
        self.parser.add_argument('-aug_mode', type=str, default='simple', choices=['simple', 'heavy', 'heavy2'], help="Augmentation mode.")
        self.parser.add_argument('-pin_memory', action='store_true', help="Use pin_memory for DataLoader.")
        self.parser.add_argument('-scratch', action='store_true', help="Transfer data to scratch space (if on SLURM).")
        self.parser.add_argument('-rev', action='store_true', help="Reverse source and target domains.")
        self.parser.add_argument('-spacing', type=float, default=1.0, help="Pixel spacing for dataset (e.g., MMWHS).")
        self.parser.add_argument('--noM3AS', action='store_false', dest='use_m3as', help="Disable M3AS style augmentation (if applicable).")
        self.parser.set_defaults(use_m3as=True) 
        self.parser.add_argument('-clahe', action='store_true', help="Apply CLAHE preprocessing.")
        self.parser.add_argument("-save_data", action='store_true', help="Save augmented/processed batch data (for debugging).")

        # Model arguments
        self.parser.add_argument("-backbone", type=str, default='resnet50', help="Encoder backbone (e.g., resnet50, drunet).")
        self.parser.add_argument("-pretrained", action='store_true', help="Use ImageNet pretrained weights for encoder.")
        self.parser.add_argument("-restore_from", type=str, default=None, help="Path to checkpoint to restore model from.")
        self.parser.add_argument("-num_classes", type=int, default=config.NUM_CLASSES, help="Number of segmentation classes.")
        self.parser.add_argument("-nb", type=int, default=4, help="Number of blocks (for DRUNet).")
        self.parser.add_argument("-bd", type=int, default=4, help="Bottleneck depth (for DRUNet).")
        self.parser.add_argument("-filters", type=int, default=32, help="Base number of filters (for DRUNet).")
        self.parser.add_argument('-multilvl', action='store_true', default=False, help="Model has multi-level outputs (default: False).")

        # Optimizer arguments
        self.parser.add_argument('-optim', type=str, default='sgd', choices=['sgd', 'adam'], help="Optimizer type.")
        self.parser.add_argument('-lr', type=float, default=config.LEARNING_RATE, help="Learning rate.")
        self.parser.add_argument("-momentum", type=float, default=config.MOMENTUM, help="SGD momentum.")
        self.parser.add_argument("-weight_decay", type=float, default=config.WEIGHT_DECAY, help="L2 Weight decay.")
        self.parser.add_argument('-lr_decay_method', type=str, default=None, choices=[None, 'poly', 'linear'], help="LR decay method.")
        self.parser.add_argument('-lr_decay', type=float, default=config.LEARNING_RATE_DECAY, help="LR decay factor (for linear).")
        self.parser.add_argument('-lr_end', type=float, default=0.0, help="Minimum LR for poly decay.")
        self.parser.add_argument("-power", type=float, default=config.POWER, help="Poly LR decay power.")
        
        self.parser.add_argument('-vgg', type=str, default=getattr(config, 'VGG_NORMALIZED_PATH', '/kaggle/input/vgg-normalised/vgg_normalised.pth'), help="Path to VGG normalized weights for RAIN.")
        self.parser.add_argument('-style_dir', type=str, default='./style_track', help="Directory to save stylized images.")
        self.parser.add_argument('-save_every_epochs', type=int, default=config.SAVE_PRED_EVERY, help="Frequency to save stylized images/checkpoints.")

        self.add_additional_arguments() 
        
        script_args = [arg for arg in sys.argv[1:] if not arg.startswith('-f')] 
        self.args = self.parser.parse_args(script_args if script_args else [])

        # Determine dataset and modalities AFTER args are parsed
        if 'mscmrseg' in self.args.data_dir.lower():
            self.dataset = 'mscmrseg'
            self.trgt_modality = 'bssfp' if self.args.rev else 'lge'
            self.src_modality = 'lge' if self.args.rev else 'bssfp'
        elif 'mmwhs' in self.args.data_dir.lower():
            self.dataset = 'mmwhs'
            if not hasattr(self.args, 'raw') or self.args.raw is None: 
                 self.args.raw = True 
            self.trgt_modality = 'ct' if self.args.rev else 'mr' 
            self.src_modality = 'ct' if not self.args.rev else 'mr'
        else:
            raise NotImplementedError(f"Dataset not recognized from data_dir: {self.args.data_dir}")
        
        self.args.spacing = 0.5 if 'DDFSeg' in self.args.data_dir else 1.0


    def _initialize_training_resources(self):
        self.get_arguments_apdx() 
        
        if self.writer is None: 
             self.writer, self.log_dir = get_summarywriter(self.apdx)
        print(f"APDX for this run: {self.apdx}") 
        print(f"Log directory: {self.log_dir}")

        self.args.num_workers = min(self.cores, self.args.bs) if self.args.num_workers == -1 else self.args.num_workers
        
        self.prepare_dataloader()
        self.prepare_model()
        self.prepare_optimizers()
        self.prepare_checkpoints() 
        self.prepare_losses()

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed) 
            cudnn.benchmark = True 
            cudnn.deterministic = False 
            cudnn.enabled = True
        
        self.prepare_grocery() 

        if self.evaluator is None: 
            self.evaluator = Evaluator(
                data_dir=self.scratch if hasattr(self, 'scratch') and self.scratch else self.args.data_dir,
                raw_data_dir=self.scratch_raw if hasattr(self, 'scratch_raw') and self.scratch_raw else self.args.raw_data_dir,
                raw=self.args.raw,
                normalization=self.args.normalization,
                dataset=self.dataset,
                args_config=self.args # <<< --- RE-ADD args_config ---
            )

    @abstractmethod
    def add_additional_arguments(self):
        pass 

    def get_basic_arguments_apdx(self, name):
        self.apdx = f"{name}.{self.dataset}.s{self.args.split}.f{self.args.fold}.v{self.args.val_num}.{self.args.backbone}"
        if hasattr(self.args, 'use_m3as') and not self.args.use_m3as:
            self.apdx += '.noM3AS'
        if hasattr(self.args, 'apdx') and self.args.apdx: 
            self.apdx += f'.{self.args.apdx}'
        if self.args.backbone == 'drunet':
            self.apdx += f".{self.args.filters}.nb{self.args.nb}.bd{self.args.bd}"
        if getattr(self.args, 'clahe', False): self.apdx += '.clahe'
        self.apdx += f'.lr{self.args.lr:.0e}'
        if self.args.lr_decay_method is not None:
            self.apdx += f'.{self.args.lr_decay_method}'
            if self.args.lr_decay_method == 'poly': self.apdx += f'.p{self.args.power}'
            elif self.args.lr_decay_method == 'linear': self.apdx += f'.decay{self.args.lr_decay}'
        if self.args.optim == 'sgd': self.apdx += f'.mmt{self.args.momentum}'
        if self.args.raw: self.apdx += '.raw'
        if self.args.percent != 100: self.apdx += f'.pct{int(float(self.args.percent))}' # Ensure percent is int for filename
        if self.args.aug_s or self.args.aug_t:
            self.apdx += '.aug'
            if self.args.aug_s: self.apdx += 's'
            if self.args.aug_t: self.apdx += 't'
            self.apdx += self.args.aug_mode.capitalize()


    @abstractmethod
    def get_arguments_apdx(self):
        pass 

    def prepare_grocery(self):
        check_bit_generator()
        print_device_info()

    @abstractmethod
    def prepare_model(self): pass
    @abstractmethod
    def prepare_optimizers(self): pass
    @abstractmethod
    def prepare_checkpoints(self, **kwargs): pass
    @abstractmethod
    def prepare_dataloader(self): pass
    @abstractmethod
    def train_epoch(self, epoch_num, **kwargs): pass
    @abstractmethod
    def train(self, **kwargs): pass 
    
    @abstractmethod 
    def eval(self, modality='target', phase='valid', toprint=None, fold=None, **kwargs): pass

    def prepare_losses(self): pass
    def stop_training(self, epoch, epoch_start_time, current_metric): 
        stop = self.check_time_elapsed(epoch, epoch_start_time)
        if hasattr(self, 'earlystop') and self.earlystop: 
             if getattr(self.args, 'estop', False) and self.earlystop.step(current_metric):
                 print("Early stopping triggered.")
                 stop = True
        return stop

    def check_time_elapsed(self, epoch, epoch_start_time, margin=30 * 60):
        epoch_time_elapsed = datetime.now() - epoch_start_time
        print(f'Epoch {epoch + 1} time elapsed: {epoch_time_elapsed}')
        self.max_epoch_time = max(epoch_time_elapsed.total_seconds(), self.max_epoch_time)
        if (datetime.now() - self.start_time).total_seconds() > self.max_duration - self.max_epoch_time - margin:
            print("Training time limit reached.")
            return True
        return False