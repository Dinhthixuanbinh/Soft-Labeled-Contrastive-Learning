"""torch import"""
import torch
import torch.nn.functional as F
from torch import nn

import os
import numpy as np
from tqdm import tqdm
from datetime import datetime

from utils.loss import dice_loss
from utils.lr_adjust import adjust_learning_rate, adjust_learning_rate_custom
from utils import timer
from trainer.Trainer_baseline import Trainer_baseline


class Trainer_DDFSeg(Trainer_baseline):
    def __init__(self):
        super().__init__()
        self.args.num_workers = min(self.args.num_workers, int(self.cores // 2))

    def add_additional_arguments(self):
        super(Trainer_DDFSeg, self).add_additional_arguments()
        self.parser.add_argument('-pool_size', help='', type=float, default=50)
        self.parser.add_argument('-ngf', help='', type=float, default=32)
        self.parser.add_argument('-ndf', help='', type=float, default=64)
        self.parser.add_argument('-lr_dis', type=float, default=2e-4)
        self.parser.add_argument('-adjust_lr_dis', action='store_true')
        self.parser.add_argument('-w_adv_t', type=float, default=1.)
        self.parser.add_argument('-w_adv_s', type=float, default=1.)
        self.parser.add_argument('-w_cyc', type=float, default=1.)
        self.parser.add_argument('-w_adv_aux', type=float, default=.1)
        self.parser.add_argument('-w_zero', type=float, default=.01)
        self.parser.add_argument('-w_seg', type=float, default=.1)
        self.parser.add_argument('-w_adv_seg', type=float, default=.1)
        self.parser.add_argument('-mmt1', help='the momentum for the discriminators', type=float, default=0.9)
        self.parser.add_argument('-mmt', help='the momentum for the discriminators', type=float, default=0.99)
        self.parser.add_argument('-restore_ddfnet', type=str, default=None)
        self.parser.add_argument('-restore_d_s', type=str, default=None)
        self.parser.add_argument('-restore_d_t', type=str, default=None)
        self.parser.add_argument('-restore_d_seg', type=str, default=None)
        self.parser.add_argument('-restore_encoderc', type=str, default=None)
        self.parser.add_argument('-restore_encodert', type=str, default=None)
        self.parser.add_argument('-restore_segdecoder', type=str, default=None)

    def get_arguments_apdx(self):
        super(Trainer_DDFSeg, self).get_basic_arguments_apdx(name='DDFSeg')

        self.apdx += f".bs{self.args.bs}"
        self.apdx += f".lr_d{self.args.lr_dis}.advT{self.args.w_adv_t}." \
                     f"S{self.args.w_adv_s}.aux{self.args.w_adv_aux}.Seg{self.args.w_adv_seg}.Cyc{self.args.w_cyc}." \
                     f"z{self.args.w_zero}"

    @timer.timeit
    def prepare_model(self):
        from model.DDFSeg import DDFNet, SegDecoder
        from model.GAN import PathGAN, PathGAN_aux
        self.segdecoder = SegDecoder()
        if self.args.restore_from:
            checkpoint = torch.load(self.args.restore_from)
            self.segdecoder.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f'DDFNet loaded from: {self.args.restore_from}')

        self.segdecoder.train()
        self.segdecoder.to(self.device)

        self.ddfnet = DDFNet()
        if self.args.restore_ddfnet:
            checkpoint = torch.load(self.args.restore_ddfnet)
            self.ddfnet.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f'DDFNet loaded from: {self.args.restore_ddfnet}')
        if self.args.restore_encoderc:
            checkpoint = torch.load(self.args.restore_encoderc)
            self.ddfnet.encoderc.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f'encoderc loaded from: {self.args.restore_encoderc}')
        if self.args.restore_encodert:
            checkpoint = torch.load(self.args.restore_encodert)
            self.ddfnet.encodert.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f'encodert loaded from: {self.args.restore_encodert}')

        self.ddfnet.train()
        self.ddfnet.to(self.device)

        self.segmentor = nn.Sequential(self.ddfnet.encoderc, self.ddfnet.encodert, self.segdecoder)

        self.d_s = PathGAN_aux()
        if self.args.restore_d_s:
            checkpoint = torch.load(self.args.restore_d_s)
            self.d_s.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f'Discriminator S loaded from: {self.args.restore_d_s}')
        self.d_s.train()
        self.d_s.to(self.device)
        print('Discriminator S created')

        self.d_t = PathGAN()
        if self.args.restore_d_t:
            checkpoint = torch.load(self.args.restore_d_t)
            self.d_t.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f'Discriminator T loaded from: {self.args.restore_d_t}')
        self.d_t.train()
        self.d_t.to(self.device)
        print('Discriminator T created')

        self.d_seg = PathGAN(input_nc=4)
        if self.args.restore_d_seg:
            checkpoint = torch.load(self.args.restore_d_seg)
            self.d_seg.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f'Discriminator Seg loaded from: {self.args.restore_d_seg}')
        self.d_seg.train()
        self.d_seg.to(self.device)
        print('Discriminator Seg created')

    @timer.timeit
    def prepare_checkpoints(self, mode='max'):
        from utils.callbacks import ModelCheckPointCallback
        weight_root_dir = './weights/'
        super(Trainer_DDFSeg, self).prepare_checkpoints(mode=mode)

        """create the discriminator checkpoints"""
        DDFNet_weight_dir = weight_root_dir + 'DDFNet_{}.pt'.format(self.apdx)
        best_DDFNet_weight_dir = weight_root_dir + 'best_DDFNet_{}.pt'.format(self.apdx)
        self.mcp_DDFNet = ModelCheckPointCallback(n_epochs=self.args.epochs, save_best=True,
                                                  mode=mode,
                                                  best_model_dir=best_DDFNet_weight_dir,
                                                  save_last_model=True,
                                                  model_name=DDFNet_weight_dir,
                                                  entire_model=False)
        segdcdr_weight_dir = weight_root_dir + 'segdcdr_{}.pt'.format(self.apdx)
        best_segdcdr_weight_dir = weight_root_dir + 'best_segdcdr_{}.pt'.format(self.apdx)
        self.mcp_segdcdr = ModelCheckPointCallback(n_epochs=self.args.epochs, save_best=True,
                                                   mode=mode,
                                                   best_model_dir=best_segdcdr_weight_dir,
                                                   save_last_model=True,
                                                   model_name=segdcdr_weight_dir,
                                                   entire_model=False)
        d_s_weight_dir = weight_root_dir + 'd_s_{}.pt'.format(self.apdx)
        best_d_s_weight_dir = weight_root_dir + 'best_d_s_{}.pt'.format(self.apdx)
        self.mcp_d_s = ModelCheckPointCallback(n_epochs=self.args.epochs, save_best=True,
                                               mode=mode,
                                               best_model_dir=best_d_s_weight_dir,
                                               save_last_model=True,
                                               model_name=d_s_weight_dir,
                                               entire_model=False)
        d_t_weight_dir = weight_root_dir + 'd_t_{}.pt'.format(self.apdx)
        best_d_t_weight_dir = weight_root_dir + 'best_d_t_{}.pt'.format(self.apdx)
        self.mcp_d_t = ModelCheckPointCallback(n_epochs=self.args.epochs, save_best=True,
                                               mode=mode,
                                               best_model_dir=best_d_t_weight_dir,
                                               save_last_model=True,
                                               model_name=d_t_weight_dir,
                                               entire_model=False)
        d_segdcdr_weight_dir = weight_root_dir + 'd_segdcdr_{}.pt'.format(self.apdx)
        best_d_segdcdr_weight_dir = weight_root_dir + 'best_d_segdcdr_{}.pt'.format(self.apdx)
        self.mcp_d_segdcdr = ModelCheckPointCallback(n_epochs=self.args.epochs, save_best=True,
                                                     mode=mode,
                                                     best_model_dir=best_d_segdcdr_weight_dir,
                                                     save_last_model=True,
                                                     model_name=d_segdcdr_weight_dir,
                                                     entire_model=False)
        print('discriminator checkpoints created')

    @timer.timeit
    def prepare_optimizers(self):
        params = self.segdecoder.parameters()
        if self.args.optim == 'sgd':
            self.opt_segdcdr = torch.optim.SGD(params, lr=self.args.lr, momentum=self.args.momentum,
                                               weight_decay=self.args.weight_decay)
        elif self.args.optim == 'adam':
            self.opt_segdcdr = torch.optim.Adam(params, lr=self.args.lr, betas=(0.9, 0.99))
        else:
            raise NotImplementedError
        if self.args.restore_from:
            checkpoint = torch.load(self.args.restore_from)
            if 'optimizer_state_dict' in checkpoint.keys():
                try:
                    self.opt_segdcdr.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Optimizer loaded from state dict: {}".format(os.path.basename(self.args.restore_from)))
                except Exception as e:
                    print(f'Error when loading the optimizer: {e}')
        self.opt_segdcdr.zero_grad()
        print('Segmentor optimizer created')

        self.opt_ddfnet = torch.optim.Adam([{'params': self.ddfnet.parameters()},
                                            {'params': self.ddfnet.encoders.att.gamma},
                                            {'params': self.ddfnet.encodert.att.gamma}], lr=self.args.lr,
                                           betas=(self.args.mmt1, self.args.mmt))
        if self.args.restore_ddfnet:
            checkpoint = torch.load(self.args.restore_ddfnet)
            try:
                self.opt_ddfnet.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer loaded from state dict: {}".format(os.path.basename(self.args.restore_ddfnet)))
            except Exception as e:
                print(f'Error when loading the optimizer: {e}')
            self.opt_ddfnet.zero_grad()
        print('DDFNet optimizer created')

        self.opt_d_s = torch.optim.Adam(self.d_s.parameters(), lr=self.args.lr_dis,
                                        betas=(self.args.mmt1, self.args.mmt))
        if self.args.restore_d_s:
            checkpoint = torch.load(self.args.restore_d_s)
            try:
                self.opt_d_s.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer loaded from state dict: {}".format(os.path.basename(self.args.restore_d_s)))
            except Exception as e:
                print(f'Error when loading the optimizer: {e}')
            self.opt_d_s.zero_grad()
        print('Source discriminator optimizer created')

        self.opt_d_t = torch.optim.Adam(self.d_t.parameters(), lr=self.args.lr_dis,
                                        betas=(self.args.mmt1, self.args.mmt))
        if self.args.restore_d_t:
            checkpoint = torch.load(self.args.restore_d_t)
            try:
                self.opt_d_t.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer loaded from state dict: {}".format(os.path.basename(self.args.restore_d_t)))
            except Exception as e:
                print(f'Error when loading the optimizer: {e}')
            self.opt_d_t.zero_grad()
        print('Target discriminator optimizer created')

        self.opt_d_seg = torch.optim.Adam(self.d_seg.parameters(), lr=self.args.lr_dis,
                                          betas=(self.args.mmt1, self.args.mmt))
        if self.args.restore_d_seg:
            checkpoint = torch.load(self.args.restore_d_seg)
            try:
                self.opt_d_seg.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer loaded from state dict: {}".format(os.path.basename(self.args.restore_d_seg)))
            except Exception as e:
                print(f'Error when loading the optimizer: {e}')
            self.opt_d_seg.zero_grad()
        print('Segmentation discriminator optimizer created')

    def adjust_lr(self, epoch):
        if self.args.lr_decay_method == 'poly':
            adjust_learning_rate(optimizer=self.opt_segdcdr, epoch=epoch, lr=self.args.lr, warmup_epochs=0,
                                 power=self.args.power,
                                 epochs=self.args.epochs)
        elif self.args.lr_decay_method == 'linear':
            adjust_learning_rate_custom(optimizer=self.opt_segdcdr, lr=self.args.lr, lr_decay=self.args.lr_decay,
                                        epoch=epoch)
        elif self.args.lr_decay_method is None:
            pass
        else:
            raise NotImplementedError
        if self.args.lr_decay_method == 'poly':
            adjust_learning_rate(optimizer=self.opt_ddfnet, epoch=epoch, lr=self.args.lr, warmup_epochs=0,
                                 power=self.args.power,
                                 epochs=self.args.epochs)
        elif self.args.lr_decay_method == 'linear':
            adjust_learning_rate_custom(optimizer=self.opt_ddfnet, lr=self.args.lr, lr_decay=self.args.lr_decay,
                                        epoch=epoch)
        elif self.args.lr_decay_method is None:
            pass
        else:
            raise NotImplementedError
        if self.args.adjust_lr_dis:
            if self.args.lr_decay_method == 'poly':
                adjust_learning_rate(self.opt_d_s, epoch, self.args.lr_dis, warmup_epochs=0,
                                     power=self.args.power, epochs=self.args.epochs)
                adjust_learning_rate(self.opt_d_s, epoch, self.args.lr_dis, warmup_epochs=0,
                                     power=self.args.power, epochs=self.args.epochs)

    def prepare_losses(self):
        super(Trainer_DDFSeg, self).prepare_losses()
        from torch import nn
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def train_epoch(self, epoch):
        print(f'start to train epoch: {epoch}')
        smooth = 1e-10
        self.segdecoder.train()
        self.ddfnet.train()
        self.d_t.train()
        self.d_s.train()
        self.d_seg.train()
        resultls = {}
        real_label = 1
        fake_label = 0
        cyc_loss_s_list, cyc_loss_t_list = [], []
        loss_seg_list, loss_seg_recon_s_list, zero_loss_s_list, zero_loss_t_list = [], [], [], []
        loss_adv_d_s_list, loss_adv_d_s_aux_list, loss_adv_d_t_list, loss_adv_pred_t_list = [], [], [], []

        loss_d_t_list, loss_d_pred_list, loss_d_recon_s_list, loss_d_s_list = [], [], [], []
        d_t_acc_real, d_t_acc_fake = [], []
        d_pred_recon_s_acc_real, d_pred_t_acc_fake = [], []
        d_s_acc_real, d_s_acc_fake = [], []
        d_recon_s_acc_real, d_s_acc_aux_fake = [], []

        for batch_content, batch_style in zip(self.content_loader, self.style_loader):
            self.opt_segdcdr.zero_grad()
            self.opt_ddfnet.zero_grad()
            self.opt_d_s.zero_grad()
            self.opt_d_t.zero_grad()
            self.opt_d_seg.zero_grad()
            for param in self.d_s.parameters():
                param.requires_grad = False
            for param in self.d_t.parameters():
                param.requires_grad = False
            for param in self.d_seg.parameters():
                param.requires_grad = False
            img_s, labels_s, names = batch_content
            img_s, labels_s = img_s.to(self.device, non_blocking=self.args.pin_memory), \
                              labels_s.to(self.device, non_blocking=self.args.pin_memory)
            img_t, labels_t, namet = batch_style
            img_t = img_t.to(self.device, non_blocking=self.args.pin_memory)

            ddf_out = self.ddfnet(img_s, img_t)

            """Segmentation losses"""
            pred_s = self.segdecoder(ddf_out['content_s'])  # (N, 4, 224, 224)
            seg_loss = self.ce_loss(pred_s, labels_s.long()) + dice_loss(pred_s, labels_s)
            loss_seg_list.append(seg_loss.item())
            pred_recon_s = self.segdecoder(ddf_out['recon_content_s'])  # (N, 4, 224, 224)
            recon_s_seg_loss = self.ce_loss(pred_recon_s, labels_s.long()) + dice_loss(pred_recon_s, labels_s)
            seg_loss += recon_s_seg_loss
            loss_seg_recon_s_list.append(recon_s_seg_loss.item())
            pred_t = self.segdecoder(ddf_out['content_t'])  # (N, 4, 224, 224)

            """Cycle consistency losses"""
            zero_loss_s = self.mse_loss(ddf_out['style_s_from_t'], torch.zeros_like(ddf_out['style_s_from_t']))
            zero_loss_t = self.mse_loss(ddf_out['style_t_from_s'], torch.zeros_like(ddf_out['style_t_from_s']))
            zero_loss_s_list.append(zero_loss_s.item())
            zero_loss_t_list.append(zero_loss_t.item())

            cyc_loss_s = self.mse_loss(ddf_out['recon_imgs'], img_s[:, 1: 2])
            cyc_loss_t = self.mse_loss(ddf_out['recon_imgt'], img_t[:, 1: 2])
            cyc_loss_s_list.append(cyc_loss_s.item())
            cyc_loss_t_list.append(cyc_loss_t.item())

            total_loss = (self.args.w_seg * seg_loss + self.args.w_cyc * (cyc_loss_s + cyc_loss_t) +
                          self.args.w_zero * (zero_loss_s + zero_loss_t))

            """Adversarial losses"""
            """fake image with source content and target style (fake target image)"""
            d_fake_st = self.d_t(ddf_out['fake_img_s_t'])  # (N, 1, 224, 224) -> (N, 1, 31, 31)
            loss_adv_d_s = F.binary_cross_entropy_with_logits(d_fake_st, torch.FloatTensor(
                d_fake_st.data.size()).fill_(
                real_label).cuda())
            loss_adv_d_t_list.append(loss_adv_d_s.item())

            # loss_adv_d_s_aux = F.binary_cross_entropy_with_logits(d_fake_st_aux, torch.FloatTensor(
            #     d_fake_st.data.size()).fill_(
            #     real_label).cuda())
            # loss_adv_d_t_aux_list.append(loss_adv_d_s_aux.item())

            d_pred_t = self.d_seg(pred_t.detach())  # (N, 4, 224, 224) -> (N, 1, 31, 31)
            loss_pred_d_t = F.binary_cross_entropy_with_logits(d_pred_t, torch.FloatTensor(
                d_pred_t.data.size()).fill_(
                real_label).cuda())
            loss_adv_pred_t_list.append(loss_pred_d_t.item())

            """fake image with source style and target content (fake source image)"""
            d_fake_ts, d_fake_ts_aux = self.d_s(
                ddf_out['fake_img_t_s'])  # (N, 1, 224, 224) -> (N, 1, 31, 31), (N, 1, 31, 31)
            loss_adv_d_t = F.binary_cross_entropy_with_logits(d_fake_ts, torch.FloatTensor(
                d_fake_ts.data.size()).fill_(
                real_label).cuda())
            loss_adv_d_s_list.append(loss_adv_d_t.item())

            loss_adv_d_t_aux = F.binary_cross_entropy_with_logits(d_fake_ts_aux, torch.FloatTensor(
                d_fake_ts.data.size()).fill_(
                real_label).cuda())
            loss_adv_d_s_aux_list.append(loss_adv_d_t_aux.item())

            total_loss += self.args.w_adv_t * loss_adv_d_s + self.args.w_adv_seg * loss_pred_d_t + \
                          self.args.w_adv_s * loss_adv_d_t + self.args.w_adv_aux * loss_adv_d_t_aux

            total_loss.backward()

            """Calculate the discriminators losses"""
            for param in self.d_s.parameters():
                param.requires_grad = True
            for param in self.d_t.parameters():
                param.requires_grad = True
            for param in self.d_seg.parameters():
                param.requires_grad = True

            """The target discriminator"""
            d_real_t = self.d_t(img_t[:, 1: 2])
            loss_d_real_t = F.binary_cross_entropy_with_logits(d_real_t, torch.FloatTensor(d_real_t.data.size()).fill_(
                real_label).cuda()) / 2
            loss_d_real_t.backward()
            d_real_t = torch.sigmoid(d_real_t.detach()).cpu().numpy()
            d_real_t = np.where(d_real_t >= .5, 1, 0)
            d_t_acc_real.append(np.mean(d_real_t))

            d_fake_t = self.d_t(ddf_out['fake_img_s_t'].detach())
            loss_d_fake_t = F.binary_cross_entropy_with_logits(d_fake_t,
                                                               torch.FloatTensor(d_fake_t.data.size()).fill_(
                                                                   fake_label).cuda()) / 2
            loss_d_fake_t.backward()
            loss_d_t_list.append((loss_d_real_t + loss_d_fake_t).item())
            d_fake_t = torch.sigmoid(d_fake_t.detach()).cpu().numpy()
            d_fake_t = np.where(d_fake_t >= .5, 1, 0)
            d_t_acc_fake.append(1 - np.mean(d_fake_t))

            """The pred discriminator"""
            d_pred_recon_s = self.d_seg(pred_recon_s.detach())
            loss_d_pred_recon_s = F.binary_cross_entropy_with_logits(d_pred_recon_s,
                                                                     torch.FloatTensor(
                                                                         d_pred_recon_s.data.size()).fill_(
                                                                         real_label).cuda()) / 2
            loss_d_pred_recon_s.backward()
            d_pred_recon_s = torch.sigmoid(d_pred_recon_s.detach()).cpu().numpy()
            d_pred_recon_s = np.where(d_pred_recon_s >= .5, 1, 0)
            d_pred_recon_s_acc_real.append(np.mean(d_pred_recon_s))

            d_pred_t = self.d_seg(pred_t.detach())
            loss_d_pred_t = F.binary_cross_entropy_with_logits(d_pred_t,
                                                               torch.FloatTensor(d_pred_t.data.size()).fill_(
                                                                   fake_label).cuda()) / 2
            loss_d_pred_t.backward()
            loss_d_pred_list.append(loss_d_pred_t.item() + loss_d_pred_recon_s.item())
            d_pred_t = torch.sigmoid(d_pred_t.detach()).cpu().numpy()
            d_pred_t = np.where(d_pred_t >= .5, 1, 0)
            d_pred_t_acc_fake.append(1 - np.mean(d_pred_t))

            """The source discriminator"""
            d_real_s, _ = self.d_s(img_s[:, 1: 2])
            loss_d_real_s = F.binary_cross_entropy_with_logits(d_real_s, torch.FloatTensor(d_real_s.data.size()).fill_(
                real_label).cuda()) / 2
            loss_d_real_s.backward()
            d_real_s = torch.sigmoid(d_real_s.detach()).cpu().numpy()
            d_real_s = np.where(d_real_s >= .5, 1, 0)
            d_s_acc_real.append(np.mean(d_real_s))

            _, d_recon_s = self.d_s(ddf_out['recon_imgs'].detach())
            loss_d_recon_s = F.binary_cross_entropy_with_logits(d_recon_s,
                                                                torch.FloatTensor(d_recon_s.data.size()).fill_(
                                                                    real_label).cuda()) / 2
            loss_d_recon_s.backward()
            d_recon_s = torch.sigmoid(d_recon_s.detach()).cpu().numpy()
            d_recon_s = np.where(d_recon_s >= .5, 1, 0)
            d_recon_s_acc_real.append(np.mean(d_recon_s))

            d_fake_s, d_fake_s_aux = self.d_s(ddf_out['fake_img_t_s'].detach())
            loss_d_fake_s = F.binary_cross_entropy_with_logits(d_fake_s,
                                                               torch.FloatTensor(d_fake_s.data.size()).fill_(
                                                                   fake_label).cuda()) / 2
            d_fake_s = torch.sigmoid(d_fake_s.detach()).cpu().numpy()
            d_fake_s = np.where(d_fake_s >= .5, 1, 0)
            d_s_acc_fake.append(1 - np.mean(d_fake_s))

            loss_d_s_list.append(loss_d_real_s.item() + loss_d_fake_s.item())

            loss_d_fake_s_aux = F.binary_cross_entropy_with_logits(d_fake_s_aux,
                                                                   torch.FloatTensor(d_fake_s_aux.data.size()).fill_(
                                                                       fake_label).cuda()) / 2
            loss_d_recon_s_list.append(loss_d_recon_s.item() + loss_d_fake_s_aux.item())
            d_fake_s_aux = torch.sigmoid(d_fake_s_aux.detach()).cpu().numpy()
            d_fake_s_aux = np.where(d_fake_s_aux >= .5, 1, 0)
            d_s_acc_aux_fake.append(1 - np.mean(d_fake_s_aux))
            (loss_d_fake_s + loss_d_fake_s_aux).backward()

            """update the discriminator optimizer"""
            self.opt_d_t.step()
            self.opt_d_s.step()
            self.opt_d_seg.step()
            """update the generator and the segDecoder optimizer"""
            self.opt_segdcdr.step()
            self.opt_ddfnet.step()

        resultls['cyc_loss_s'] = sum(cyc_loss_s_list) / len(cyc_loss_s_list)
        resultls['cyc_loss_t'] = sum(cyc_loss_t_list) / len(cyc_loss_t_list)
        resultls['seg_s'] = sum(loss_seg_list) / len(loss_seg_list)
        resultls['seg_fake_st'] = sum(loss_seg_recon_s_list) / len(loss_seg_recon_s_list)
        resultls['zero_loss_s'] = sum(zero_loss_s_list) / len(zero_loss_s_list)
        resultls['zero_loss_t'] = sum(zero_loss_t_list) / len(zero_loss_t_list)

        resultls['loss_adv_s'] = sum(loss_adv_d_s_list) / len(loss_adv_d_s_list)
        resultls['loss_adv_s_aux'] = sum(loss_adv_d_s_aux_list) / len(loss_adv_d_s_aux_list)
        resultls['loss_adv_t'] = sum(loss_adv_d_t_list) / len(loss_adv_d_t_list)
        resultls['loss_adv_pred_t'] = sum(loss_adv_pred_t_list) / len(loss_adv_pred_t_list)

        resultls['loss_d_s'] = sum(loss_d_s_list) / len(loss_d_s_list)
        resultls['loss_d_recon_s'] = sum(loss_d_recon_s_list) / len(loss_d_recon_s_list)
        resultls['loss_d_t'] = sum(loss_d_t_list) / len(loss_d_t_list)
        resultls['loss_d_pred'] = sum(loss_d_pred_list) / len(loss_d_pred_list)

        resultls['acc_t_real'] = sum(d_t_acc_real) / len(d_t_acc_real)
        resultls['acc_t_fake'] = sum(d_t_acc_fake) / len(d_t_acc_fake)
        resultls['acc_pred_recon_s_real'] = sum(d_pred_recon_s_acc_real) / len(d_pred_recon_s_acc_real)
        resultls['acc_pred_t_fake'] = sum(d_pred_t_acc_fake) / len(d_pred_t_acc_fake)
        resultls['acc_s_real'] = sum(d_s_acc_real) / len(d_s_acc_real)
        resultls['acc_s_fake'] = sum(d_s_acc_fake) / len(d_s_acc_fake)
        resultls['acc_s_recon_real'] = sum(d_recon_s_acc_real) / len(d_recon_s_acc_real)
        resultls['acc_s_aux_fake'] = sum(d_s_acc_aux_fake) / len(d_s_acc_aux_fake)

        return resultls

    def train(self):
        """
        :return:
        """
        for epoch in tqdm(range(self.start_epoch, self.args.epochs)):
            epoch_start = datetime.now()
            """adjust learning rate"""
            self.adjust_lr(epoch)

            train_results = self.train_epoch(epoch)
            self.segmentor = nn.Sequential(self.ddfnet.encoderc, self.ddfnet.encodert, self.segdecoder).to(self.device)
            results = self.eval(phase='valid')
            lge_dice = np.round((results['dc'][0] + results['dc'][2] + results['dc'][4]) / 3, 3)
            if self.args.evalT:
                results = self.eval(modality='target', phase='test')
                lge_dice_test = np.round((results['dc'][0] + results['dc'][2] + results['dc'][4]) / 3, 3)

            """record all the experiment results into the tensorboard"""
            print("Writing summary")
            if self.args.evalT:
                self.writer.add_scalars('Dice/LGE', {'Valid': lge_dice, 'Test': lge_dice_test}, epoch + 1)
            else:
                self.writer.add_scalar('Dice/LGE_valid', lge_dice, epoch + 1)
            """record accuracies"""
            self.writer.add_scalars('Acc/T_Dis', {'real': train_results['acc_t_real'],
                                                  'fake': train_results['acc_t_fake']}, epoch + 1)
            self.writer.add_scalars('Acc/Seg_Dis', {'Recon': train_results['acc_pred_recon_s_real'],
                                                    'target': train_results['acc_pred_t_fake']}, epoch + 1)
            self.writer.add_scalars('Acc/S_Dis', {'real': train_results['acc_s_real'],
                                                  'fake': train_results['acc_s_fake']}, epoch + 1)
            self.writer.add_scalars('Acc/S_Dis_aux', {'Recon': train_results['acc_s_recon_real'],
                                                      'fake': train_results['acc_s_aux_fake']}, epoch + 1)
            # if self.args.multilvl:
            """record losses"""
            self.writer.add_scalars('Loss/Seg', {'main': train_results['seg_s'], 'Recon': train_results['seg_fake_st']},
                                    epoch + 1)
            self.writer.add_scalars('Loss/Cycle', {'s': train_results['cyc_loss_s'], 't': train_results['cyc_loss_t']},
                                    epoch + 1)
            self.writer.add_scalars('Loss/Zero', {'s': train_results['zero_loss_s'], 't': train_results['zero_loss_t']},
                                    epoch + 1)
            self.writer.add_scalars('Loss/Adv', {'s': train_results['loss_adv_s'],
                                                 's_aux': train_results['loss_adv_s_aux'],
                                                 't': train_results['loss_adv_t'],
                                                 'seg': train_results['loss_adv_pred_t']}, epoch + 1)
            self.writer.add_scalars('Loss/S_Dis', {'s': train_results['loss_d_s'],
                                                   's_recon': train_results['loss_d_recon_s']}, epoch + 1)
            self.writer.add_scalar('Loss/T_Dis', train_results['loss_d_t'], epoch + 1)
            self.writer.add_scalar('Loss/Seg_DIs', train_results['loss_d_pred'], epoch + 1)
            self.writer.add_scalars('LR', {'Seg': self.opt_segdcdr.param_groups[0]['lr'],
                                           'Dis': self.opt_d_t.param_groups[0]['lr']}, epoch + 1)

            print(
                f'Epoch = {epoch + 1:4d}/{self.args.epochs:4d}, loss_seg = {train_results["seg_s"]:.4f}, dc_valid = {lge_dice:.4f}')

            tobreak = self.stop_training(epoch, epoch_start, lge_dice)

            self.mcp_segdcdr.step(monitor=lge_dice, model=self.segdecoder, epoch=epoch + 1,
                                  optimizer=self.opt_segdcdr,
                                  tobreak=tobreak)
            self.mcp_DDFNet.step(monitor=lge_dice, model=self.ddfnet, epoch=epoch + 1,
                                 optimizer=self.opt_ddfnet,
                                 tobreak=tobreak)
            # if self.args.multilvl:
            self.mcp_d_t.step(monitor=lge_dice, model=self.d_t, epoch=epoch + 1,
                              optimizer=self.opt_d_t,
                              tobreak=tobreak)
            self.mcp_d_s.step(monitor=lge_dice, model=self.d_s, epoch=epoch + 1,
                              optimizer=self.opt_d_s,
                              tobreak=tobreak)
            self.mcp_d_segdcdr.step(monitor=lge_dice, model=self.d_seg, epoch=epoch + 1,
                                    optimizer=self.opt_d_seg,
                                    tobreak=tobreak)

            self.mcp_segmentor.step(monitor=lge_dice, model=self.segmentor, epoch=epoch + 1,
                                    tobreak=tobreak)
            if tobreak:
                break

        self.writer.close()
        best_epoch = self.mcp_segmentor.epoch
        best_score = self.mcp_segmentor.best_result
        log_dir = 'runs/{}.e{}.Scr{}'.format(self.apdx, best_epoch,
                                             np.around(best_score, 3))
        os.rename(self.log_dir, log_dir)
        # load the weights with the bext validation score and do the evaluation
        print("the weight of the best segmentor model: {}".format(
            self.mcp_segmentor.best_model_save_dir))

        """test the model with the test data"""
        try:
            self.segmentor.load_state_dict(torch.load(
                self.mcp_segmentor.best_model_save_dir)['model_state_dict'])
            print("segmentor load from state dict")
        except:
            self.segmentor.load_state_dict(torch.load(
                self.mcp_segmentor.best_model_save_dir))
        print("Segmentor loaded")
        self.segmentor = nn.Sequential(self.ddfnet.encoderc, self.ddfnet.encodert, self.segdecoder).to(self.device)
        self.eval(modality='target', phase='test')
        return
