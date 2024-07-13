from datetime import datetime
import os
import numpy as np
from tqdm import tqdm

import torch

from dataset.data_generator_mscmrseg import prepare_dataset
from dataset.data_generator_mmwhs import prepare_dataset as prepare_dataset_mmwhs
from utils.lr_adjust import adjust_learning_rate, adjust_learning_rate_custom
from utils import timer
import config
from utils.loss import loss_calc
from trainer.Trainer import Trainer


class Trainer_baseline(Trainer):
    def __init__(self):
        super().__init__()

    def add_additional_arguments(self):
        """
        :param parser:
        :return:
        """

        """dataset configuration"""
        self.parser.add_argument("-train_with_t", action='store_true')
        self.parser.add_argument("-train_with_s", action='store_true')
        """evaluation configuration"""
        self.parser.add_argument("-eval_bs", type=int, default=config.EVAL_BS,
                                 help="Number of images sent to the network in a batch during evaluation.")
        self.parser.add_argument('-toggle_klc',
                                 help='Whether to apply keep_largest_component in evaluation during training.',
                                 action='store_false')
        self.parser.add_argument('-hd95', action='store_true')
        self.parser.add_argument('-multilvl', help='if apply multilevel network', action='store_true')

    @timer.timeit
    def get_arguments_apdx(self):
        """
        :return:
        """
        assert self.args.train_with_s or self.args.train_with_t, "at least train on one domain."

        super(Trainer_baseline, self).get_basic_arguments_apdx(name='Base')
        self.apdx += f".bs{self.args.bs}.aug_{self.args.aug_mode}"
        self.apdx += '.trainW'
        if self.args.train_with_s:
            self.apdx += 's'
        if self.args.train_with_t:
            self.apdx += 't'
        if self.args.normalization == 'zscore':
            self.apdx += '.zscr'
        elif self.args.normalization == 'minmax':
            self.apdx += '.mnmx'
        print(f'apdx: {self.apdx}')

    @timer.timeit
    def prepare_dataloader(self):
        if self.dataset == 'mscmrseg':
            self.scratch, self.scratch_raw, self.content_loader, self.style_loader = prepare_dataset(self.args)
        elif self.dataset == 'mmwhs':
            self.scratch, self.scratch_raw, self.content_loader, self.style_loader = prepare_dataset_mmwhs(self.args)
        else:
            raise NotImplementedError

    @timer.timeit
    def prepare_model(self):
        if self.args.backbone == 'unet':
            from model.unet_model import UNet
            self.segmentor = UNet(n_channels=3, n_classes=self.args.num_classes)
        elif self.args.backbone == 'drunet':
            from model.DRUNet import Segmentation_model as DR_UNet
            self.segmentor = DR_UNet(filters=self.args.filters, n_block=self.args.nb, bottleneck_depth=self.args.bd,
                                     n_class=self.args.num_classes, multilvl=self.args.multilvl)
            if self.args.restore_from:
                checkpoint = torch.load(self.args.restore_from)
                try:
                    self.segmentor.load_state_dict(checkpoint['model_state_dict'], strict=True)
                except:
                    self.segmentor.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif self.args.backbone == 'deeplabv2':
            from model.deeplabv2 import get_deeplab_v2
            self.segmentor = get_deeplab_v2(num_classes=self.args.num_classes, multi_level=self.args.multilvl,
                                            input_size=224)
            if self.args.restore_from:
                checkpoint = torch.load(self.args.restore_from)
                if self.args.pretrained:
                    new_params = self.segmentor.state_dict().copy()
                    for i in checkpoint:
                        i_parts = i.split('.')
                        if not i_parts[1] == 'layer5':
                            new_params['.'.join(i_parts[1:])] = checkpoint[i]
                    self.segmentor.load_state_dict(new_params)
                else:
                    self.segmentor.load_state_dict(checkpoint['model_state_dict'])
        elif 'resnet' in self.args.backbone or 'efficientnet' in self.args.backbone or \
                'mobilenet' in self.args.backbone or 'densenet' in self.args.backbone or 'ception' in self.args.backbone or \
                'se_resnet' in self.args.backbone or 'skresnext' in self.args.backbone:
            from model.segmentation_models import segmentation_models
            self.segmentor = segmentation_models(name=self.args.backbone, pretrained=False,
                                                 decoder_channels=(512, 256, 128, 64, 32), in_channel=3,
                                                 classes=4, multilvl=self.args.multilvl)
            if self.args.restore_from:
                checkpoint = torch.load(self.args.restore_from)
                try:
                    self.segmentor.load_state_dict(checkpoint['model_state_dict'], strict=True)
                    print('model loaded strict')
                except:
                    self.segmentor.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print('model loaded no strict')
            elif self.args.pretrained:
                from utils.utils_ import get_pretrained_checkpoint
                checkpoint = get_pretrained_checkpoint(self.args.backbone)
                self.segmentor.encoder.load_state_dict(checkpoint)
        else:
            raise NotImplementedError

        if self.args.restore_from and (not self.args.pretrained) and 'epoch' in checkpoint.keys():
            try:
                self.start_epoch = self.start_epoch if self.args.pretrained else checkpoint['epoch']
            except Exception as e:
                self.start_epoch = 0
                print(f'Error when loading the epoch number: {e}')

        self.segmentor.train()
        self.segmentor.to(self.device)

    @timer.timeit
    def prepare_checkpoints(self, mode='max'):
        from utils.callbacks import ModelCheckPointCallback
        weight_root_dir = './weights/'
        if not os.path.exists(weight_root_dir):
            os.mkdir(weight_root_dir)
        weight_dir = os.path.join(weight_root_dir, self.apdx + '.pt')
        best_weight_dir = os.path.join(weight_root_dir, "best_" + self.apdx + '.pt')
        # create the model check point
        self.mcp_segmentor = ModelCheckPointCallback(n_epochs=self.args.epochs, save_best=True,
                                                     mode=mode,
                                                     best_model_dir=best_weight_dir,
                                                     save_last_model=True,
                                                     model_name=weight_dir,
                                                     entire_model=False)
        print('model checkpoint created')

    @timer.timeit
    def prepare_optimizers(self):
        if self.args.backbone == 'deeplabv2':
            params = self.segmentor.optim_parameters(self.args.lr)
        # self.args.backbone == 'drunet' or ('resnet' in self.args.backbone)
        else:
            params = self.segmentor.parameters()
        if self.args.optim == 'sgd':
            self.opt = torch.optim.SGD(params, lr=self.args.lr, momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optim == 'adam':
            self.opt = torch.optim.Adam(params, lr=self.args.lr, betas=(0.9, 0.99))
        else:
            raise NotImplementedError
        if self.args.restore_from:
            checkpoint = torch.load(self.args.restore_from)
            if 'optimizer_state_dict' in checkpoint.keys():
                try:
                    self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Optimizer loaded from state dict: {}".format(os.path.basename(self.args.restore_from)))
                except Exception as e:
                    print(f'Error when loading the optimizer: {e}')
        self.opt.zero_grad()
        print('Segmentor optimizer created')

    def adjust_lr(self, epoch):
        if self.args.lr_decay_method == 'poly':
            adjust_learning_rate(optimizer=self.opt, epoch=epoch, lr=self.args.lr, warmup_epochs=0,
                                 power=self.args.power,
                                 epochs=self.args.epochs)
        elif self.args.lr_decay_method == 'linear':
            adjust_learning_rate_custom(optimizer=self.opt, lr=self.args.lr, lr_decay=self.args.lr_decay,
                                        epoch=epoch)
        elif self.args.lr_decay_method is None:
            pass
        else:
            raise NotImplementedError

    def eval(self, modality='target', phase='valid', toprint=None):
        if phase == 'valid':
            results = self.evaluator.evaluate_single_dataset(seg_model=self.segmentor, ifhd=False, ifasd=False,
                                                             modality=self.trgt_modality if modality == 'target' else self.src_modality,
                                                             phase=phase, bs=self.args.eval_bs, toprint=True if toprint is None else toprint,
                                                             klc=self.args.toggle_klc, crop_size=self.args.crop, spacing=self.args.spacing)
        elif phase == 'test':
            results = self.evaluator.evaluate_single_dataset(seg_model=self.segmentor,
                                                             modality=self.trgt_modality if modality == 'target' else self.src_modality,
                                                             phase=phase, spacing=self.args.spacing,
                                                             ifhd=True, toprint=True if toprint is None else toprint,
                                                             ifhd95=self.args.hd95, ifasd=True, save_csv=False,
                                                             weight_dir=None, klc=True if self.dataset == 'mscmrseg' else False,
                                                             bs=self.args.eval_bs,
                                                             lge_train_test_split=None, crop_size=self.args.crop,
                                                             pred_index=0, fold_num=self.args.fold, split=self.args.split)
        else:
            raise NotImplementedError
        return results

    def train_epoch(self, **kwargs):
        pass

    @timer.timeit
    def train(self):
        self.eval(modality='target', phase='test')


if __name__ == '__main__':
    trainer_base = Trainer_baseline()
    trainer_base.train()
    print('program finished')
