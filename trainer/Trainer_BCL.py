from datetime import datetime
import os
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
import yaml
from cv2 import imwrite

import torch
from torch.nn import functional as F

from dataset.data_generator_mscmrseg import init_test_dataset
from utils.lr_adjust import adjust_learning_rate, adjust_learning_rate_custom
from utils.utils_ import easy_dic, mkdir, show_config, thres_cb_plabel, gene_plabel_prop, mask_fusion, Acc
from utils import timer
import config
from trainer.Trainer_baseline import Trainer_baseline
from utils.metrics import *


class Trainer_BCL(Trainer_baseline):
    def __init__(self):
        super().__init__()
        if self.args.config is not None:
            with open(self.args.config, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            config = easy_dic(config)
            config.snapshot = os.path.join(config.snapshot, config.note)
            mkdir(config.snapshot)
            print('Snapshot stored in: {}'.format(config.snapshot))
            """merge arguments with the configuration, so that it is easier to retrieve all the configurations"""
            self.args = easy_dic(vars(self.args))
            config.update(self.args)
            self.args = config
            message = show_config(config)
            print(message)
            self.start_round = 0

    def add_additional_arguments(self):
        """
        :param parser:
        :return:
        """
        super(Trainer_BCL, self).add_additional_arguments()
        self.parser.add_argument('-config', help='the path to the configuration file.', default=None)
        self.parser.add_argument('-lambt', help='the lambda weight for the target segmentation loss.', default=.3)
        self.parser.add_argument('-lamb', help='the lambda weight for the target segmentation loss.', default=.4)
        self.parser.add_argument('-round', help='The maximum number of rounds.', type=int, default=10)
        self.parser.add_argument('-cb_prop', help='the basic probability', default=.1)
        self.parser.add_argument('-thres_inc', help='the increment of the threshold.', default=0)

    @timer.timeit
    def get_arguments_apdx(self):
        """
        :return:
        """
        super(Trainer_BCL, self).get_basic_arguments_apdx(name='BCL')

    @timer.timeit
    def prepare_dataloader(self):
        super(Trainer_BCL, self).prepare_dataloader()
        if self.dataset == 'mscmrseg':
            self.init_target = init_test_dataset(self.args, self.scratch)

    def prepare_losses(self):
        from utils.loss import loss_entropy_BCL, bidirect_contrastive_loss_BCL
        self.LossEntropy = loss_entropy_BCL
        self.metric_loss = bidirect_contrastive_loss_BCL

    @timer.timeit
    def prepare_model(self):
        from model.BCL_DeeplabV2 import ResPair_Deeplab
        self.segmentor = ResPair_Deeplab(4)
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

            if (not self.args.pretrained) and 'epoch' in checkpoint.keys():
                try:
                    self.start_epoch = self.start_epoch if self.args.pretrained else checkpoint['epoch']
                except Exception as e:
                    self.start_epoch = 0
                    print(f'Error when loading the epoch number: {e}')
            if (not self.args.pretrained) and 'round' in checkpoint.keys():
                try:
                    self.start_round = self.start_round if self.args.pretrained else checkpoint['round']
                except Exception as e:
                    self.start_round = 0
                    print(f'Error when loading the epoch number: {e}')

        self.segmentor.train()
        self.segmentor.to(self.device)

    def save_pred(self, round_):
        """
        Generate the global and local thresholds. Use the threshold to generate pseudo labels and save the pseudo label.
        Average accuracy and the proportion of the pseudo labels are calculated as well.
        :param round_: the round number
        :return:
        """
        print("[Generate pseudo labels]")
        interp = torch.nn.Upsample(size=(self.args.crop, self.args.crop), mode="bilinear", align_corners=True)

        self.plabel_path = os.path.join(self.args.plabel, self.args.note, str(round_))

        mkdir(self.plabel_path)
        self.args.target_data_dir = self.plabel_path
        # save the probability of pseudo labels for the pixel-wise similarity matching, which is detailed around Eq. (9)
        # self.pool = Pool()
        accs = AverageMeter()  # Counter
        props = AverageMeter()  # Counter
        cls_acc = GroupAverageMeter()  # Class-wise Acc/Prop of Pseudo labels
        with torch.no_grad():
            for index, batch in tqdm(enumerate(self.init_target)):
                image, label, _, _, name = batch
                label = label.to(self.device)
                img_name = name[0].split("/")[-1]
                dir_name = name[0].split("/")[0]
                img_name = img_name.replace("leftImg8bit", "gtFine_labelIds")
                temp_dir = os.path.join(self.plabel_path, dir_name)
                if not os.path.exists(temp_dir):
                    os.mkdir(temp_dir)

                output, _ = self.segmentor.forward(image.to(self.device), source=False)
                output = interp(output)  # (1, C, H, W)
                # the mask and the pseudo labels selected by global threshold
                mask, plabel = thres_cb_plabel(output, self.cb_thres, num_cls=self.args.num_classes)
                # the mask and the pseudo labels selected by local threshold
                mask2, plabel2 = gene_plabel_prop(output, self.args.cb_prop)
                # fuse the global and local mask and generate the pseudo label with the mask
                # The fusion strategy is detailed in Sec. 3.3 of paper
                mask, plabel = mask_fusion(output, mask, mask2)  # (H, W), (H, W)
                # update the average ratio (between the sum of the probability and the total number of selected pixel)
                # WARNING: Bug in the calculation of the average ratio
                # self.pool.update_pool(output, mask=mask.float())
                # NOTE:
                #   acc: the accuracy for the overall predictions that is over the threshold. sensitivity: TP / TP + FN
                #   prop: the proportion of the selected pixels whose prediction is larger than the threshold.
                #   acc_dict: {class_index: (accuracy, total number of prediction of the class)}
                acc, prop, cls_dict = Acc(plabel, label, num_cls=self.args.num_classes)
                cnt = (plabel != 255).sum().item()
                accs.update(acc, cnt)
                props.update(prop, 1)
                cls_acc.update(cls_dict)
                plabel = plabel.view(1024, 2048)
                plabel = plabel.cpu().numpy()

                plabel = np.asarray(plabel, dtype=np.uint8)
                save_path = "%s/%s.png" % (temp_dir, img_name.split(".")[0])
                imwrite(save_path, plabel)

        print('The Accuracy :{:.2%} and proportion :{:.2%} of Pseudo Labels'.format(accs.avg.item(), props.avg.item()))
        # if self.config.neptune:
        #     neptune.send_metric("Acc", accs.avg)
        #     neptune.send_metric("Prop", props.avg)

    def gene_thres(self, prop, num_cls=19):  # prop = 0.1
        """
        Do the predict, and collect all the probabilities correspond to each class. Find the thresholds that pick out
        the top :prop ratio predictions of each class respectively
        :param prop: default .1
        :param num_cls: number of classes
        :return: the thresholds that pick out the top :prop ratio predictions of each class respectively
        """
        print('[Calculate Threshold using config.cb_prop]')  # r in section 3.3

        probs = {}  # store a dictionary for the probability prediction of each class
        for index, batch in tqdm(enumerate(self.init_target)):
            img, label, _, _, _ = batch  # img (1, 3, H, W)
            with torch.no_grad():
                # x1, _ = self.model.forward(img.to(self.device))
                x1, _ = self.segmentor.forward(img.to(self.device), source=False)  # x1 (1, C, h, w)
                pred = F.softmax(x1, dim=1)  # (1, c, h, w)get the softmax prediction
            pred_probs = pred.max(dim=1)[0]  # (h, w) get the max pred value for each pixel
            pred_probs = pred_probs.squeeze()  # (h, w)
            pred_label = torch.argmax(pred, dim=1).squeeze()  # (h, w) the predicted class indices
            for i in range(num_cls):
                cls_mask = pred_label == i  # get the mask for the class
                cnt = cls_mask.sum()  # number of pixels assigned to the class
                if cnt == 0:
                    continue  # when no pixel is assigned to the class
                cls_probs = torch.masked_select(pred_probs, cls_mask)  # pick out the predictions belonging to the class
                cls_probs = cls_probs.detach().cpu().numpy().tolist()
                cls_probs.sort()  # from the smallest to the largest (necessary as it will be down-sampled)
                if i not in probs:
                    probs[i] = cls_probs[::5]  # reduce the consumption of memory
                else:  # probs = {0 (class number): [probs of each prediction], 1: [], 2: [] ...}
                    probs[i].extend(cls_probs[::5])  # collect all the class pixels throughout the batches

        thres = {}
        for k in probs.keys():  # key represents the class index
            cls_prob = probs[k]
            cls_total = len(cls_prob)
            cls_prob = np.array(cls_prob)
            cls_prob = np.sort(cls_prob)
            index = int(cls_total * prop)
            cls_thres = cls_prob[-index]  # the threshold that split the top prop values
            cls_thres2 = cls_prob[index]
            # if cls_thres == 1.0:
            #    cls_thres = 0.999
            thres[k] = cls_thres  # store the threshold for each class index
        if self.args.source == 'synthia':
            thres[9] = 1
            thres[14] = 1
            thres[16] = 1
        # for i in range(self.config.num_classes):
        #    if i in thres:
        #        continue
        #    else:
        #        thres[i] = 1
        print(thres)
        return thres

    def train_epoch(self, epoch):
        print(f'start to train epoch: {epoch}')
        results = EasyDict({'loss_source':[], 'loss_target': [], 'loss_entropy': [], 'loss_metric': []})

        for batch_source, batch_target in zip(self.content_loader, self.style_loader):
            self.segmentor.train()
            self.opt.zero_grad()
            img_s, label_s, names = batch_source
            img_t, label_t, namet = batch_target

            pred_s, feature_s = self.segmentor.forward(img_s.cuda())
            pred_t, feature_t = self.segmentor.forward(img_t.cuda())

            label_s = label_s.long().to(self.device)
            label_t = label_t.long().to(self.device)

            loss_s = F.cross_entropy(pred_s, label_s, ignore_index=255)
            loss_t = F.cross_entropy(pred_t, label_t, ignore_index=255)
            loss_seg = (loss_s + self.args.lambt * loss_t)

            loss_e = self.LossEntropy(pred_s).mean() + self.args.lambt * self.LossEntropy(pred_t).mean()
            loss_e = self.args.lamb * loss_e
            # TODO: make it compatible with multi batch size DONE:)
            loss_metric = 0
            counter = 0
            for label_s_, feature_s_, label_t_, feature_t_ in zip(label_s, feature_s, label_t, feature_t):
                """extract the label and the feature correspond to one source sample"""
                label_s_ = label_s_.unsqueeze(0).contiguous()
                feature_s_ = feature_s_.unsqueeze(0).contiguous()
                """extract the label and the feature correspond to one target sample"""
                label_t_ = label_t_.unsqueeze(0).contiguous()
                feature_t_ = feature_t_.unsqueeze(0).contiguous()
                loss_metric_ = self.metric_loss(feature_s_, label_s_, feature_t_, label_t_, self.args.num_classes, self.args)
                loss_metric += loss_metric_
                counter += 1

            loss_metric = loss_metric / counter
            loss = loss_seg + loss_e + loss_metric

            results.loss_source = results.loss_source.append(loss_s.item())
            results.loss_target = results.loss_target.append(loss_t.item())
            results.loss_entropy = results.loss_entropy.append(loss_e.item())
            results.loss_metric = results.loss_metric.append(loss_metric.item())
            loss.backward()
            self.opt.step()

        results.loss_source = sum(results.loss_source) / len(results.loss_source)
        results.loss_target = sum(results.loss_target) / len(results.loss_target)
        results.loss_entropy = sum(results.loss_entropy) / len(results.loss_target)
        results.loss_metric = sum(results.loss_metric) / len(results.loss_metric)

        return results

    @timer.timeit
    def train(self):
        for r in range(self.start_round, self.args.round):
            self.cb_thres = self.gene_thres(
                self.args.cb_prop + self.args.thres_inc * r)  # thresholds for all the classes
            self.save_pred(r)
            self.plabel_path = os.path.join(self.args.plabel, self.args.note, str(r))

            self.prepare_optimizers()
            for epoch in tqdm(range(self.start_epoch, self.args.epochs)):
                """adjust learning rate and save the value for tensorboard"""
                self.adjust_lr(epoch=epoch)
                epoch_start = datetime.now()

                train_results = self.train_epoch(epoch)

                msg = f'Epoch = {epoch + 1:6d}/{self.args.epochs:6d}'
                if self.args.train_with_s:
                    msg += f', loss_seg = {train_results["seg_s"]:.4f}'
                if self.args.train_with_t:
                    msg += f', loss_seg_t = {train_results["seg_t"]: .4f}'
                print(msg)
                results = self.eval(modality='target', phase='valid')
                lge_dice = np.round((results['dc'][0] + results['dc'][2] + results['dc'][4]) / 3, 3)
                if self.args.evalT:
                    results = self.eval(modality='target', phase='test')
                    lge_dice_test = np.round((results['dc'][0] + results['dc'][2] + results['dc'][4]) / 3, 3)
                    self.writer.add_scalars('Dice/LGE', {'Valid': lge_dice, 'Test': lge_dice_test}, epoch + 1)
                else:
                    self.writer.add_scalar('Dice/LGE_valid', lge_dice, epoch + 1)

                tobreak = self.check_time_elapsed(epoch, epoch_start)

                self.mcp_segmentor.step(monitor=lge_dice, model=self.segmentor, epoch=epoch + 1,
                                        optimizer=self.opt,
                                        tobreak=tobreak)

                if self.args.train_with_s:
                    if self.args.train_with_t:
                        self.writer.add_scalars('Loss Seg',
                                                {'Source': train_results['seg_s'], 'Target': train_results['seg_t']},
                                                epoch + 1)
                    else:
                        self.writer.add_scalar('Loss Seg/Source', train_results['seg_s'], epoch + 1)
                else:
                    self.writer.add_scalar('Loss Seg/Target', train_results['seg_t'], epoch + 1)
                self.writer.add_scalar('LR/Seg_LR', self.opt.param_groups[0]['lr'], epoch + 1)

                if tobreak:
                    break
            self.args.lr = self.args.lr / 2  # decay the learning rate for each round

        self.writer.close()
        best_epoch = self.mcp_segmentor.epoch
        best_score = self.mcp_segmentor.best_result
        log_dir_new = 'runs/{}.e{}.Scr{}'.format(self.apdx, best_epoch,
                                                 np.around(best_score, 3))
        os.rename(self.log_dir, log_dir_new)
        # load the weights with the bext validation score and do the evaluation
        print("the weight of the best unet model: {}".format(self.mcp_segmentor.best_model_save_dir))
        try:
            self.segmentor.load_state_dict(torch.load(self.mcp_segmentor.best_model_save_dir)['model_state_dict'])
            print("segmentor load from state dict")
        except:
            self.segmentor.load_state_dict(torch.load(self.mcp_segmentor.best_model_save_dir))
        print("model loaded")
        self.eval(modality='target', phase='test')
        if self.args.train_with_s:
            self.eval(modality='target', phase='test', fold=1 - self.args.fold)
        self.eval(modality='source', phase='test')
        return
