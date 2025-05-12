# %%writefile /kaggle/working/Soft-Labeled-Contrastive-Learning/trainer/Trainer.py
import torch.backends.cudnn as cudnn
import torch

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import os
import argparse
import random
import numpy as np

from utils.utils_ import get_device, check_bit_generator, print_device_info, get_summarywriter
import config
from evaluator import Evaluator


class Trainer(ABC):
    def __init__(self):
        self.start_time = datetime.now()
        self.max_epoch_time = 0
        self.start_epoch = 0
        self.max_duration = 24 * 3600 - 5 * 60 # Approx 24 hours minus 5 minutes
        self.device = get_device()
        self.cores = os.cpu_count()

        self.get_argparser() # Parse initial command-line arguments and defaults

        # Initialize attributes that will be set later
        self.apdx = None
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
        self.parser.add_argument('-apdx', type=str, default='')
        self.parser.add_argument('-evalT', action='store_true')
        self.parser.add_argument('-val_num', type=int, default=0)
        self.parser.add_argument('-spacing', help='pixel spacing', type=float, default=1.0) # More generic
        self.parser.add_argument('-noM3AS', action='store_false', default=True) # Default is M3AS disabled
        self.parser.add_argument('-data_dir', type=str, default=config.DATA_DIRECTORY)
        self.parser.add_argument("-raw_data_dir", type=str, default=config.RAW_DATA_DIRECTORY)
        self.parser.add_argument('-rev', action='store_true')
        self.parser.add_argument('-fold', type=int, default=0)
        self.parser.add_argument('-split', type=int, default=0)
        self.parser.add_argument('-scratch', action='store_true')
        self.parser.add_argument('-bs', type=int, default=config.BATCH_SIZE)
        self.parser.add_argument('-crop', type=int, default=config.INPUT_SIZE)
        self.parser.add_argument('-aug_s', action='store_true')
        self.parser.add_argument('-aug_t', action='store_true')
        self.parser.add_argument('-aug_mode', type=str, default='simple')
        self.parser.add_argument('-pin_memory', action='store_true')
        self.parser.add_argument('-num_workers', type=int, default=-1)
        self.parser.add_argument('-normalization', type=str, default='minmax')
        self.parser.add_argument('-clahe', action='store_true')
        self.parser.add_argument("-raw", action='store_true')
        self.parser.add_argument("-percent", type=float, default=100)
        self.parser.add_argument("-save_data", action='store_true')
        self.parser.add_argument("-backbone", help='model backbone', type=str, default='resnet50') # Default to resnet50
        self.parser.add_argument("-pretrained", action='store_true', help="use ImageNet pretrained encoder")
        self.parser.add_argument("-restore_from", type=str, default=None)
        self.parser.add_argument("-num_classes", type=int, default=config.NUM_CLASSES)
        self.parser.add_argument("-nb", type=int, default=4, help="N_block for DRUNet")
        self.parser.add_argument("-bd", type=int, default=4, help="Bottleneck depth for DRUNet")
        self.parser.add_argument("-filters", type=int, default=32, help="Filters for DRUNet")
        self.parser.add_argument('-optim', type=str, default='sgd')
        self.parser.add_argument('-lr_decay_method', type=str, default=None)
        self.parser.add_argument('-lr', type=float, default=config.LEARNING_RATE)
        self.parser.add_argument('-lr_decay', type=float, default=config.LEARNING_RATE_DECAY)
        self.parser.add_argument('-lr_end', type=float, default=0)
        self.parser.add_argument("-momentum", type=float, default=config.MOMENTUM)
        self.parser.add_argument("-power", type=float, default=config.POWER)
        self.parser.add_argument("-weight_decay", type=float, default=config.WEIGHT_DECAY)
        self.parser.add_argument('-epochs', type=int, default=config.EPOCHS)
        self.parser.add_argument('-vgg', type=str, default='/kaggle/input/vgg-normalised/vgg_normalised.pth')
        self.parser.add_argument('-style_dir', type=str, default='./style_track')
        self.parser.add_argument('-save_every_epochs', type=int, default=config.SAVE_PRED_EVERY)
        self.parser.add_argument("-seed", type=int, default=config.RANDOM_SEED)

        self.add_additional_arguments() # Allow child classes to add their specific arguments
        
        # Parse arguments - use empty list for notebooks if not passing actual command line args
        # Child script (like train_MCCL.py) will override these if it modifies self.args AFTER Trainer init
        current_args = sys.argv[1:] if hasattr(sys, 'argv') and len(sys.argv) > 1 else []
        self.args = self.parser.parse_args(current_args if current_args else [])


        if 'mmwhs' in self.args.data_dir.lower():
            self.args.raw = True # Ensure raw is used for MMWHS if not explicitly set
            self.dataset = 'mmwhs'
            self.trgt_modality = 'ct' if self.args.rev else 'mr'
            self.src_modality = 'ct' if not self.args.rev else 'mr'
        elif 'mscmrseg' in self.args.data_dir.lower():
            self.dataset = 'mscmrseg'
            self.trgt_modality = 'bssfp' if self.args.rev else 'lge'
            self.src_modality = 'lge' if self.args.rev else 'bssfp'
        else:
            raise NotImplementedError(f"Dataset not recognized from data_dir: {self.args.data_dir}")
        
        self.args.spacing = 0.5 if 'DDFSeg' in self.args.data_dir else 1.0
        # del self.parser # Keep parser if child classes need to inspect it, or del later

    def _initialize_training_resources(self):
        """
        Called at the beginning of the train() method in child classes
        AFTER self.args might have been modified by hardcoding in the launch script.
        """
        self.get_arguments_apdx() # Now uses the potentially modified self.args
        
        if self.writer is None: # Initialize writer only once
             self.writer, self.log_dir = get_summarywriter(self.apdx)

        self.args.num_workers = min(self.cores, self.args.bs) if self.args.num_workers == -1 else self.args.num_workers
        
        self.prepare_dataloader()
        self.prepare_model()
        self.prepare_optimizers()
        self.prepare_checkpoints() # Relies on self.apdx
        self.prepare_losses()

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)

        if self.evaluator is None: # Initialize evaluator only once
            self.evaluator = Evaluator(data_dir=self.scratch if hasattr(self, 'scratch') and self.scratch else self.args.data_dir,
                                   raw_data_dir=self.scratch_raw if hasattr(self, 'scratch_raw') and self.scratch_raw else self.args.raw_data_dir,
                                   raw=self.args.raw,
                                   normalization=self.args.normalization,
                                   dataset=self.dataset)
        self.prepare_grocery()


    @abstractmethod
    def add_additional_arguments(self):
        pass

    def get_basic_arguments_apdx(self, name):
        # Ensure all components of apdx are derived from the potentially modified self.args
        self.apdx = f"{name}.{self.dataset}.s{self.args.split}.f{self.args.fold}.v{self.args.val_num}.{self.args.backbone}"
        if hasattr(self.args, 'noM3AS') and not self.args.noM3AS:
            self.apdx += '.noM3AS'
        if hasattr(self.args, 'apdx') and self.args.apdx != '':
            self.apdx += f'.{self.args.apdx}'
        if self.args.backbone == 'drunet':
            self.apdx += f".{self.args.filters}.nb{self.args.nb}.bd{self.args.bd}"
        if getattr(self.args, 'clahe', False):
            self.apdx += '.clahe'
        self.apdx += f'.lr{self.args.lr}'
        if self.args.lr_decay_method is not None:
            self.apdx += f'.{self.args.lr_decay_method}'
            if self.args.lr_decay_method == 'linear':
                self.apdx += f'.decay{self.args.lr_decay}'
            if self.args.lr_decay_method == 'poly':
                self.apdx += f'.power{self.args.power}'
        if self.args.optim == 'sgd':
            self.apdx += f'.mmt{self.args.momentum}'
        if 'DDFSeg' in self.args.data_dir:
            self.apdx += '.res.5'
        if self.args.raw:
            self.apdx += '.raw'
            if self.args.percent != 100:
                self.apdx += f'.pct{self.args.percent}'
        if self.args.aug_s or self.args.aug_t:
            self.apdx += '.aug'
            if self.args.aug_s: self.apdx += 's'
            if self.args.aug_t: self.apdx += 't'
            if self.args.aug_mode == 'simple': self.apdx += 'Sim'
            elif '2' in self.args.aug_mode: self.apdx += 'Hvy2'
            elif self.args.aug_mode == 'heavy': self.apdx += 'Hvy'

    @abstractmethod
    def get_arguments_apdx(self): # Implemented by child classes
        pass

    def prepare_grocery(self): # Called once from __init__
        check_bit_generator()
        print_device_info()
        cudnn.benchmark = True
        cudnn.enabled = True
        # torch.autograd.set_detect_anomaly(True) # Keep disabled for performance unless debugging specific NaN issues

    @abstractmethod
    def prepare_model(self): pass
    @abstractmethod
    def prepare_optimizers(self): pass
    @abstractmethod
    def prepare_checkpoints(self, **kwargs): pass
    @abstractmethod
    def prepare_dataloader(self): pass
    @abstractmethod
    def train_epoch(self, **kwargs): pass
    @abstractmethod
    def train(self, **kwargs): pass # Child class implements the training loop
    @abstractmethod
    def eval(self, phase='valid'): pass
    def prepare_losses(self): pass
    def stop_training(self, *args): pass

    def check_time_elapsed(self, epoch, epoch_start, margin=30 * 60):
        epoch_time_elapsed = datetime.now() - epoch_start
        print(f'Epoch {epoch + 1} time elapsed: {epoch_time_elapsed}')
        self.max_epoch_time = max(epoch_time_elapsed.seconds, self.max_epoch_time)
        if (datetime.now() - self.start_time).seconds > self.max_duration - self.max_epoch_time - margin:
            print("Training time limit reached.")
            return True
        return False