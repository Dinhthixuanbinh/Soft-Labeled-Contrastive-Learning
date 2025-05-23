# %%writefile /kaggle/working/Soft-Labeled-Contrastive-Learning/utils/utils_.py
from torch.nn import init
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import center_crop

from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import sys 
import os
import re
import nibabel as nib
from skimage import measure # Ensure skimage.measure is imported
import argparse
from pathlib import Path
from glob import glob
# Assuming timer.py is in utils directory or accessible via 'from utils import timer'
# If timer is a sibling file: import timer
# If utils is a package and timer is a module in it: from . import timer
# Based on your project structure, utils is a package.
from utils import timer 

import SimpleITK as sitk
from easydict import EasyDict

from utils.callbacks import ModelCheckPointCallback # Assuming callbacks.py is in utils
import config # Assuming config.py is at the project root


def print_device_info():
    if torch.cuda.is_available():
        print("Device name: {}".format(torch.cuda.get_device_name(0)))
        print("torch version: {}".format(torch.__version__))
        print("device count: {}".format(torch.cuda.device_count()))
        # The following line is redundant if the first one works.
        # print('device name: {}'.format(torch.cuda.get_device_name(0))) 
    else:
        print("CUDA is not available. Using CPU.")
    print(f'Number of cores: {os.cpu_count()}')

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_summarywriter(log_subdir=''):
    from torch.utils.tensorboard import SummaryWriter
    # Ensure 'runs' directory exists
    base_runs_dir = Path('runs')
    base_runs_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir_path = base_runs_dir / log_subdir
    
    # Check if the exact log_subdir path exists. If so, append timestamp.
    # This prevents errors if a non-timestamped log_subdir is re-used.
    if log_dir_path.exists() and not any(char.isdigit() for char in log_dir_path.name.split('.')[-1]): # Heuristic for non-timestamped
        now = datetime.now()
        # Append timestamp to the *specific subdirectory name*, not the whole path string
        log_dir_path = base_runs_dir / f"{log_subdir}.{now.strftime('%H.%M.%S')}" # More unique timestamp
        
    log_dir_path.mkdir(parents=True, exist_ok=True) # Ensure it exists before writer uses it
    
    writer = SummaryWriter(log_dir=str(log_dir_path))
    return writer, str(log_dir_path)


def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


def read_img(pat_id, img_len, file_path='../processed/', modality='lge'):
    images = []
    if modality == 'bssfp':
        folder = 'testA' if pat_id < 6 else 'trainA'
    else:
        folder = 'testB' if pat_id < 6 else 'trainB'
    modality_str = 'bSSFP' if modality == 'bssfp' else 'lge' # Renamed variable
    for im_idx in range(img_len): # Renamed variable
        img_full_path = os.path.join(file_path, "{}/pat_{}_{}_{}.png".format(folder, pat_id, modality_str, im_idx))
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Warning: Could not read image {img_full_path}")
            continue
        images.append(img)
    return np.array(images)


def keep_largest_connected_components(mask, num_total_classes):
    """
    Keeps only the largest connected components of each label for a segmentation mask.
    Args:
        mask: the image to be processed [B, C, ...]
        channel_first: true: the channel is at the 1st dimension; false: last dimension
        num_channel: to specify the number of classes
        cuda: whether to use gpu to compute the connected components
    Returns:

    """
    if mask.ndim == 2: # This line should have the first level of indent within the function
        out_img = np.zeros_like(mask, dtype=mask.dtype)
        # Iterate through foreground classes (assuming class 0 is background)
        for struc_id in range(1, num_total_classes): # Processes classes 1, 2, ..., num_total_classes-1
            binary_img_slice = (mask == struc_id)
            if np.sum(binary_img_slice) > 0: # Check if the class exists in the mask
                blobs = measure.label(binary_img_slice, connectivity=1) # Find connected components
                props = measure.regionprops(blobs)
                if not props: # No components found
                    continue
                
                areas = [ele.area for ele in props]
                if not areas: continue # Should not happen if props is not empty

                largest_blob_ind = np.argmax(areas)
                largest_blob_label = props[largest_blob_ind].label
                
                out_img[blobs == largest_blob_label] = struc_id
        return out_img
    else:
        # If you need to support batched or channel-first 3D/4D masks, add specific logic here.
        # For now, this function is tailored for the 2D `pred_slice` from evaluator.py
        raise ValueError(f"keep_largest_connected_components expects a 2D mask (H,W), but got shape {mask.shape}")
    # --- END CORRECTED INDENTATION ---
def resize_volume(img_volume, w=288, h=288):
    """
    :param img_volume:
    :return:
    """
    img_res = []
    for im in img_volume:
        img_res.append(cv2.resize(im, dsize=(w, h), interpolation=cv2.INTER_AREA))

    return np.array(img_res)


def crop_volume(vol, crop_size=112, centroid=None):
    """
    :param vol:
    :return:
    """
    if centroid is None:
        centroid = [int(vol.shape[1] / 2), int(vol.shape[2] / 2)]
    return np.array(vol[:,
                    centroid[0] - crop_size: centroid[0] + crop_size,
                    centroid[1] - crop_size: centroid[1] + crop_size, ])


def reconstruct_volume(vol, crop_size=112, origin_size=256):
    """
    :param vol:
    :return:
    """
    recon_vol = np.zeros((len(vol), 4, origin_size, origin_size), dtype=np.float32)

    recon_vol[:, :,
    int(recon_vol.shape[2] / 2) - crop_size: int(recon_vol.shape[2] / 2) + crop_size,
    int(recon_vol.shape[3] / 2) - crop_size: int(recon_vol.shape[3] / 2) + crop_size] = vol

    return recon_vol


def reconstruct_volume_torch(vol, crop_size=112, origin_size=256):
    """
    :param vol:
    :return:
    """
    recon_vol = torch.zeros((vol.size()[0], 4, origin_size, origin_size), dtype=torch.float32)

    recon_vol[:, :,
    int(recon_vol.size()[2] / 2) - crop_size: int(recon_vol.size()[2] / 2) + crop_size,
    int(recon_vol.size()[3] / 2) - crop_size: int(recon_vol.size()[3] / 2) + crop_size] = vol

    return recon_vol


def calc_mean_std(feat, eps=1e-5):
    """
    Calculate channel-wise mean and standard deviation for the input features and preserve the dimensions
    Args:
        feat: the latent feature of shape [B, C, H, W]
        eps: a small value to prevent calculation error of variance

    Returns:
    Channel-wise mean and standard deviation of the input features
    """
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization_with_noise(content_feat, style_feat):
    """
    Implementation of AdaIN of style transfer
    Args:
        content_feat: the content features of shape [N, 512, 28, 28]
        style_feat: the statistics of the style feature [M, 1024]

    Returns:
    The re-normalized features
    :param style_feat: (M, 1024)
    :param content_feat: (N, 512, 28, 28)
    """
    size = content_feat.size()
    style_mean = style_feat[:, :512].view(style_feat.size()[0], size[1], 1, 1)  # (N, 512, 1, 1)
    style_std = style_feat[:, 512:].view(style_feat.size()[0], size[1], 1, 1)  # (N, 512, 1, 1)
    content_mean, content_std = calc_mean_std(content_feat)  # (N, 512, 1, 1), (N, 512, 1, 1)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)  # (N, 512, 28, 28)
    style_std = style_std.expand(size)
    style_mean = style_mean.expand(size)
    normalized_feat = normalized_feat * style_std + style_mean  # (M * N, 512, 28, 28)
    return normalized_feat


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def calc_feat_mean_std(input, eps=1e-5):
    """
    Calculate channel-wise mean and standard deviation for the input features but reduce the dimensions
    Args:
        input: the latent feature of shape [B, C, H, W]
        eps: a small value to prevent calculation error of variance

    Returns:
    Channel-wise mean and standard deviation of the input features
    """
    size = input.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = input.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C)
    feat_mean = input.view(N, C, -1).mean(dim=2).view(N, C)
    return torch.cat([feat_mean, feat_std], dim=1)


def get_checkpoints(apdx, n_epochs, mode='min', save_every_epochs=0, decoder_best_model_dir=None,
                    decoder_model_dir=None):
    """
    Generate model checkpoints for the models in RAIN
    Args:
        decoder_model_dir: the directory to the weights of the decoder
        decoder_best_model_dir: the directory to the best weights of the decoder
        apdx: the identifier (also appendix for the files) for each weight
        n_epochs: number of epochs
        mode:
        save_every_epochs:

    Returns:

    """
    decoder_best_model_dir = 'weights/best_decoder.{}.pt'.format(
        apdx) if decoder_best_model_dir is None else decoder_best_model_dir
    decoder_model_dir = 'weights/decoder.{}.pt'.format(apdx) if decoder_model_dir is None else decoder_model_dir
    decoder_checkpoint = ModelCheckPointCallback(n_epochs=n_epochs, save_best=True,
                                                 mode=mode,
                                                 best_model_dir=decoder_best_model_dir,
                                                 save_last_model=True,
                                                 model_name=decoder_model_dir,
                                                 entire_model=False,
                                                 save_every_epochs=save_every_epochs)

    fc_encoder_best_dir = 'weights/best_fc_encoder.{}.pt'.format(apdx)
    fc_encoder_dir = 'weights/fc_encoder.{}.pt'.format(apdx)
    fc_encoder_checkpoint = ModelCheckPointCallback(n_epochs=n_epochs, save_best=True,
                                                    mode=mode,
                                                    best_model_dir=fc_encoder_best_dir,
                                                    save_last_model=True,
                                                    model_name=fc_encoder_dir,
                                                    entire_model=False,
                                                    save_every_epochs=save_every_epochs)

    fc_decoder_best_dir = 'weights/best_fc_decoder.{}.pt'.format(apdx)
    fc_decoder_dir = 'weights/fc_decoder.{}.pt'.format(apdx)
    fc_decoder_checkpoint = ModelCheckPointCallback(n_epochs=n_epochs, save_best=True,
                                                    mode=mode,
                                                    best_model_dir=fc_decoder_best_dir,
                                                    save_last_model=True,
                                                    model_name=fc_decoder_dir,
                                                    entire_model=False,
                                                    save_every_epochs=save_every_epochs)

    return decoder_checkpoint, fc_encoder_checkpoint, fc_decoder_checkpoint


def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=config.MODEL,
                        help="available options : ResNet")
    parser.add_argument('--baseline', help='whether to train a baseline (without any fancy technique).',
                        action='store_true')
    parser.add_argument("--backbone", type=str, default="deeplab",
                        help="available options: deeplab, dr-unet")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--s_aug_off", help='Whether not to apply augmentation for the source images.',
                        action='store_false')
    parser.add_argument("--target_bs", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--eval_bs", type=int, default=config.EVAL_BS,
                        help="Number of images sent to the network in a batch during evaluation.")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--raw-data-dir", type=str, default=config.RAW_DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data_shuffle", help='whether to shuffle when generate the data', action='store_true')
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-s", type=float, default=config.LEARNING_RATE_EPS,
                        help="Base learning rate for epsilon(sampling).")
    parser.add_argument("--ctd_mmt", type=float, default=0.95, help='The momentum of the source centroid.')
    parser.add_argument("--momentum", type=float, default=config.MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=config.NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=config.EPOCHS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=config.NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--warmup-steps", type=int, default=config.WARMUP_EPOCHS,
                        help="Number of training steps for early stopping.")
    parser.add_argument('--update_eps', help='Whether to update eps (sampling) in the RAIN', action='store_true')
    parser.add_argument("--eps_iters", type=int, default=config.EPS_ITERS,
                        help="Number of iterations for each epsilon. Will be used if only 'update_eps' is true")
    parser.add_argument("--power", type=float, default=config.POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-seed", type=int, default=config.RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore_from", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=config.SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--style_dir", type=str, default='style_track',
                        help="Where to save style images of the model.")
    parser.add_argument("--save-pic", help='whether to save the transferred images', action='store_true')
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")
    parser.add_argument('--vgg_encoder', type=str, default='pretrained/vgg_normalised.pth')
    parser.add_argument("--mode", help="whether train the model in 'fewshot', 'oneshot' or 'fulldata'", type=str,
                        default='oneshot')
    parser.add_argument('--vgg_decoder', type=str, default='pretrained/decoder_iter_100000.pth')
    parser.add_argument('--style_encoder', type=str, default='pretrained/fc_encoder_iter_100000.pth')
    parser.add_argument('--style_decoder', type=str, default='pretrained/fc_decoder_iter_100000.pth')
    parser.add_argument('--fp16', action='store_true',
                        help='use float16 instead of float32, which will save about 50% memory')
    parser.add_argument('--jac', help='whether to apply jaccard loss', action='store_true')
    parser.add_argument('--clda', help='whether to apply contrastive loss', action='store_true')
    parser.add_argument('--clbg', help='whether to include background in contrastive loss', action='store_true')
    parser.add_argument('--ctslv', help='Whether to apply contrastive loss between stylized S and T or S and T',
                        type=str, default=config.CONTRASTIVE_LOSS_V)
    parser.add_argument('--clwn', help='Whether to use norm as the denominator in the  contrastive loss',
                        action='store_true')
    parser.add_argument('--blockctsl', help='whether to apply block contrastive loss on stlyzied S and T',
                        default='block')
    parser.add_argument('--localctsl',
                        help='Whether to apply local contrastive loss between stylized S and T or S and T',
                        type=str, default=config.CONTRASTIVE_LOSS_V)
    parser.add_argument('--inter_w', help='the weight for the inter contrastive loss', type=float,
                        default=config.WEIGHT_INTER_LOSS)
    parser.add_argument('--intra', help='Whether to apply intra contrastive loss.', action='store_true')
    parser.add_argument('--intra_w', help='the weight for the intra contrastive loss.', type=float,
                        default=config.WEIGHT_INTRA_LOSS)
    # parser.add_argument('--inst', action='store_true')
    # parser.add_argument('--inst_w', type=float, default=config.WEIGHT_INST_LOSS)
    parser.add_argument('--mse', help='whether to apply l2 regularization to the centroids.', action='store_true')
    # parser.add_argument('--mse0', help='whether to set label of mse loss as 0.', action='store_true')
    parser.add_argument('--mse_w', help='The weight for the l2 regularization.', type=float, default=config.WEIGHT_MSE)
    parser.add_argument('--eps_cts', help='whether to apply contrastive loss to the update of epsilon.',
                        action='store_true')
    parser.add_argument('--eps_cts_w', help='The weight for the contrastive loss in the updates of epsilon.',
                        type=float, default=config.WEIGHT_EPS_CTS)
    parser.add_argument('--max_train_time', help='Set the maximum time (hours) for training.', type=int, default=None)
    parser.add_argument('--optim', help='The optimizer.', type=str, default='sgd')
    parser.add_argument('--poly', help='whether to adjust the learning rate with poly.', action='store_true')
    parser.add_argument('--scratch', help='whether is running in the cluster', action='store_true')
    parser.add_argument('--consist_w', help='the weight for the consistent loss', type=float,
                        default=config.WEIGHT_CONSIST)
    # pat_id 10, 13, 33, 38, 41
    parser.add_argument('--pat_id', help='The patient id to choose.', type=int, default=10)
    # slice_id 13, 11, 14, 7, 3
    parser.add_argument('--slice_id', help='The slice id in the volume.', type=int, default=13)
    parser.add_argument('--evl_s', help='Whether to evaluate the model in source domain.', action='store_true')
    parser.add_argument('--eval', help='Whether to evaluate the best model at the end of training.',
                        action='store_true')
    parser.add_argument('--toggle_klc', help='Whether to apply keep_largest_component in evaluation during training.',
                        action='store_false')
    parser.add_argument('--part', help='number of partitions to split decoder_ft', type=int, default=1)
    parser.add_argument('--hd95', action='store_true')
    parser.add_argument('--grad', help='Whether to record gradient of the features and the centroids',
                        action='store_true')
    parser.add_argument('--thd', help='The threshold for calculating the centroids.', type=float, default=None)
    parser.add_argument('--thd_w', help='The weight for the adaptive threshold.', type=float, default=config.WEIGHT_THD)
    parser.add_argument('--wtd_ave', help='Whether to calculated the weighted average of the features or the "global" '
                                          'average of the features as the centroids.', action='store_true')

    # ------- Sulaiman's versoin--------
    parser.add_argument('--ctslv_miccai', help='Whether to apply contrastive loss between stylized S and T or S and T',
                        action='store_true')
    return parser.parse_args()


def get_apdx_FUDA(args):
    apdx = "DR_UNet." + args.mode + '.lr{}'.format(args.learning_rate) + '.eps{}.LSeg'.format(
        args.eps_iters) + '.lrs{}'.format(args.learning_rate_s)
    if args.mode == 'fewshot':
        apdx += ".pat_10_lge"
    elif args.mode == 'oneshot':
        apdx += ".pat_10_lge_13"
    return apdx


def get_apdx(args):
    apdx = "DR_UNet." + args.mode + '.lr{}'.format(args.learning_rate) + '.cw{}'.format(args.consist_w)
    if args.ctd_mmt != 0.95:
        apdx += '.mmt{}'.format(args.ctd_mmt)
    if args.poly:
        apdx += '.poly'
    if args.update_eps:
        apdx += '.eps{}.LSeg'.format(args.eps_iters) + '.lrs{}'.format(args.learning_rate_s)
        if args.eps_cts:
            apdx += '.epcts.w{}'.format(args.eps_cts_w)
    if args.mode == 'fewshot':
        apdx += ".pat_{}_lge".format(args.pat_id)
    elif args.mode == 'oneshot':
        apdx += ".pat_{}_lge_{}".format(args.pat_id, args.slice_id)
    if args.clda:
        if args.thd is not None:
            apdx += '.thd{}.{}'.format(args.thd, args.thd_w)
        apdx += '.clda.itew{}'.format(args.inter_w)
        if args.ctslv_miccai:
            apdx += '.miccai'
        if args.intra:
            apdx += '.itrw{}'.format(args.intra_w)
        # if args.inst:
        #     apdx += '.instw{}'.format(args.inst_w)
        if args.wtd_ave:
            apdx += '.w_ave'
        if args.clbg:
            apdx += '.bg'
    if args.mse:
        apdx += '.mse.w{}'.format(args.mse_w)
    apdx += '.{}'.format(args.optim)
    apdx += '.p{}'.format(args.part)
    apdx += '.bs{}.tbs{}'.format(args.batch_size, args.target_bs)
    return apdx


def check_bit_generator():
    try:
        np.random.bit_generator = np.random._bit_generator
        print("rename numpy.random._bit_generator")
    except:
        print("numpy.random.bit_generator exists")


def cal_centroid(decoder_ft, label, previous_centroid=None, momentum=0.95, pseudo_label=False, n_class=4, partition=1,
                 threshold: int = None, thd_w: float = 0.0, weighted_ave=False, epoch=0, max_epoch=1000, # Used config.WEIGHT_THD before
                 low_thd = 0, high_thd=0.99, stdmin=False):
    """
    For source samples, previous
    :param partition: number of partitions of the centroid (only used for target features)
    :param decoder_ft: (N, 32, 256, 256)
    :param label: (N, 4, 256, 256)
    :param previous_centroid: None or (4, 32)
    :param momentum: for moving average (centroid = previous_centroid * momentum + (1 - momentum) * centroid)
    :param pseudo_label: the soft prediction of the original target images
    :param n_class: number of classes
    :param threshold: The threshold to mask out the uncertain pixels
    :param thd_w: The weight when calculating the adaptive threshold
    :param weighted_ave: Whether to calculate weighted average of the features as the centroids
    :return: the centroid of each class, ratio (the ratio of the number of features larger than the threshold)
    """
    shape_ft = decoder_ft.size()
    shape_label = label.size()
    label_temp = label
    if not pseudo_label and (shape_label[-1] != shape_ft[-1] or shape_label[-2] != shape_ft[-2]):
        if label.ndim == 3: label_temp = torch.unsqueeze(label_temp, dim=1)
        label_temp = F.interpolate(label_temp.float(), size=(shape_ft[-2], shape_ft[-1]), mode='nearest').long()
        label_temp = torch.squeeze(label_temp, dim=1)
    if pseudo_label and (shape_label[-1] != shape_ft[-1] or shape_label[-2] != shape_ft[-2]):
        if label.ndim == 3: raise ValueError("Soft pseudo-label must have channel dimension K")
        label_temp = F.interpolate(label_temp, size=(shape_ft[-2], shape_ft[-1]), mode='bilinear', align_corners=False)
    
    ratio = None; stddevs = []; current_centroids_list = []
    stddevs = []  # bg, myo, lv, rv
    if pseudo_label:
            pred_onehot = F.one_hot(torch.argmax(label_temp, dim=1), num_classes=n_class).permute(0, 3, 1, 2).float()
            pixel_certainty_mask = torch.ones_like(label_temp[:,0:1,:,:])
            if threshold is not None:
                pred_max_probs, _ = torch.max(label_temp, dim=1, keepdim=True)
                if 0 < threshold < 1: pixel_certainty_mask = (pred_max_probs >= threshold).float()

            for k_cls in range(n_class):
                if weighted_ave:
                    class_probs = label_temp[:, k_cls, :, :].unsqueeze(1) * pixel_certainty_mask
                    weighted_ft = decoder_ft * class_probs
                    sum_weighted_ft = torch.sum(weighted_ft, dim=(0, 2, 3))
                    sum_weights = torch.sum(class_probs, dim=(0, 2, 3)).squeeze() + 1e-7
                    centroid = sum_weighted_ft / sum_weights
                    current_partition_centroids_list_inner.append(centroid.unsqueeze(0))
                else: # hard pseudo labels
                    class_mask_onehot = pred_onehot[:, k_cls, :, :].unsqueeze(1) * pixel_certainty_mask
                    masked_ft = decoder_ft * class_mask_onehot
                    sum_ft = torch.sum(masked_ft, dim=(0,2,3))
                    num_pixels = torch.sum(class_mask_onehot, dim=(0,2,3)).squeeze() + 1e-7
                    current_partition_centroids_list_inner.append((sum_ft / num_pixels).unsqueeze(0))
            current_centroids_list.append(torch.cat(current_partition_centroids_list_inner, dim=0))

    else:
            current_partition_centroids_src = []
            for k_cls in range(n_class):
                class_mask = (label_temp == k_cls).unsqueeze(1).float()
                masked_ft = decoder_ft * class_mask
                sum_ft = torch.sum(masked_ft, dim=(0,2,3))
                num_pixels = torch.sum(class_mask, dim=(0,2,3)).squeeze() + 1e-7
                current_partition_centroids_src.append(sum_ft / num_pixels)
            current_centroids_list.append(torch.stack(current_partition_centroids_src, dim=0))

    final_centroids_stack = torch.stack(current_centroids_list, dim=0)
    
    if partition == 1 and pseudo_label:
        final_centroids = final_centroids_stack.squeeze(0)
    elif partition > 1 and pseudo_label:
        final_centroids = [final_centroids_stack[p] for p in range(partition)] # Return list for partition > 1
    else: # partition == 1 and not pseudo_label
        final_centroids = final_centroids_stack.squeeze(0)


    if previous_centroid is not None:
        # Handle EMA carefully if final_centroids is a list (for partition > 1)
        if isinstance(final_centroids, list) and isinstance(previous_centroid, list) and len(final_centroids) == len(previous_centroid):
            for i in range(len(final_centroids)):
                if final_centroids[i].shape == previous_centroid[i].shape:
                    final_centroids[i] = momentum * previous_centroid[i] + (1 - momentum) * final_centroids[i]
                else:
                    print(f"Warning: Shape mismatch in EMA for partition {i}")
        elif isinstance(final_centroids, torch.Tensor) and isinstance(previous_centroid, torch.Tensor) and final_centroids.shape == previous_centroid.shape:
             final_centroids = momentum * previous_centroid + (1 - momentum) * final_centroids
        else:
            print(f"Warning: Shape/type mismatch for EMA. Current type: {type(final_centroids)}, Prev type: {type(previous_centroid)}")
            
    return final_centroids, ratio, stddevs # stddevs needs to be computed if stdmin=True


def update_class_center_iter(cla_src_feas, batch_src_labels, class_center_feas, m=.2, num_class=4):
    '''
    batch_src_feas  : n*c*h*w
    barch_src_labels: n*h*w
    '''
    batch_src_feas = cla_src_feas.detach()
    batch_src_labels = batch_src_labels.cuda()
    # n, c, fea_h, fea_w = batch_src_feas.size()
    # batch_y_downsample = label_downsample(batch_src_labels, fea_h, fea_w)  # n*fea_h*fea_w
    # batch_y_downsample = batch_y_downsample.unsqueeze(1)  # n*1*fea_h*fea_w
    batch_y_downsample = batch_src_labels.unsqueeze(1)  # n*1*fea_h*fea_w
    batch_class_center_fea_list = []
    for i in range(num_class):
        fea_mask = torch.eq(batch_y_downsample, i).float().cuda()  # n*1*fea_h*fea_w
        class_feas = batch_src_feas * fea_mask  # n*c*fea_h*fea_w
        class_fea_sum = torch.sum(class_feas, [0, 2, 3])  # c
        class_num = torch.sum(fea_mask, [0, 1, 2, 3])
        if class_num == 0:
            batch_class_center_fea = class_center_feas[i, :].detach()
        else:
            batch_class_center_fea = class_fea_sum / class_num
        batch_class_center_fea = batch_class_center_fea.unsqueeze(0)  # 1 * c
        batch_class_center_fea_list.append(batch_class_center_fea)
    batch_class_center_feas = torch.cat(batch_class_center_fea_list, dim=0)  # n_class * c
    class_center_feas = m * class_center_feas + (1 - m) * batch_class_center_feas

    return class_center_feas


def generate_pseudo_label(cla_feas_trg, class_centers, pixel_sel_th=.25):
    '''
    class_centers: C*N_fea
    cla_feas_trg: N*N_fea*H*W
    '''

    def pixel_selection(batch_pixel_cosine, th):
        one_tag = torch.ones([1]).float().cuda()
        zero_tag = torch.zeros([1]).float().cuda()

        batch_sort_cosine, _ = torch.sort(batch_pixel_cosine, dim=1)
        pixel_sub_cosine = batch_sort_cosine[:, -1] - batch_sort_cosine[:, -2]
        pixel_mask = torch.where(pixel_sub_cosine > th, one_tag, zero_tag)

        return pixel_mask

    cla_feas_trg_de = cla_feas_trg.detach()
    batch, N_fea, H, W = cla_feas_trg_de.size()
    cla_feas_trg_de = F.normalize(cla_feas_trg_de, p=2, dim=1)
    class_centers_norm = F.normalize(class_centers, p=2, dim=1)
    cla_feas_trg_de = cla_feas_trg_de.transpose(1, 2).contiguous().transpose(2, 3).contiguous()  # N*H*W*N_fea
    cla_feas_trg_de = torch.reshape(cla_feas_trg_de, [-1, N_fea])
    class_centers_norm = class_centers_norm.transpose(0, 1)  # N_fea*C
    batch_pixel_cosine = torch.matmul(cla_feas_trg_de, class_centers_norm)  # N*N_class
    pixel_mask = pixel_selection(batch_pixel_cosine, pixel_sel_th)
    hard_pixel_label = torch.argmax(batch_pixel_cosine, dim=1)

    return hard_pixel_label, pixel_mask


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-7)) / np.log2(c)


def generate_dataframe(centroid_list, columns, extra_columns, i_iter, domain: str):
    arr_tosave = np.round(np.mean(np.array(centroid_list), 0), 5)
    try:
        df_tosave = pd.DataFrame(data=arr_tosave, columns=columns)
        extra_df = pd.DataFrame(data=[[1, i_iter, domain], [2, i_iter, domain], [3, i_iter, domain]],
                                columns=extra_columns)
        df_tosave = pd.concat([df_tosave, extra_df], axis=1)
    except ValueError:
        print(len(centroid_list))
        print(arr_tosave)
        print(columns)
        raise ValueError
    return df_tosave


def generate_centroid_log_path(apdx):
    path = Path('report/centroid_log/{}'.format(apdx))
    if path.exists():
        now = datetime.now()
        return generate_centroid_log_path(apdx + "_{}_{}".format(now.hour, now.minute))
    else:
        path.mkdir(parents=True)
        return path


@timer.timeit
def tranfer_data_2_scratch(data_dir, scratch):
    """
    copy the whole directory args.data_dir to /scratch/{jobid}/
    :param args:
    :param start:
    :return:
    """
    if scratch:
        jobid = os.environ['SLURM_JOB_ID']
        print(jobid)
        print('{} exists? {}'.format(Path('/scratch').joinpath(jobid), Path('/scratch').joinpath(jobid).exists()))
        scratch = Path('/scratch').joinpath(jobid).joinpath(Path(data_dir).name)
        if not scratch.exists():
            scratch.mkdir(parents=True)
            # data_from = os.path.join(data_dir, '.')
            command = 'cp -a {} {}'.format(Path(data_dir).joinpath('*'), scratch)
            print(command)
            os.system(command)
            assert scratch.exists(), f'scratch: {scratch}'
            print(glob(str(scratch.joinpath('*'))))
            print(f'Data saved in {scratch}')
        else:
            print('Data already transferred.')
    else:
        scratch = data_dir
    return str(scratch)


def get_centroids_df():
    import copy
    columns = [str(i) for i in range(32)]
    extra_columns = ['#class', '#epoch', 'domain']
    full_columns = copy.deepcopy(columns)
    full_columns.extend(extra_columns)
    centroid_pd_s = pd.DataFrame(columns=full_columns)
    centroid_pd_s_style = pd.DataFrame(columns=full_columns)
    centroid_pd_t = pd.DataFrame(columns=full_columns)
    return centroid_pd_s, centroid_pd_s_style, centroid_pd_t, columns, extra_columns


def save_transferred_images(images_s_style, apdx, args, names, target_name, i_iter, time_iter, i, idx_to_save=(0,)):
    for idx in idx_to_save:
        s_name = names[idx] if type(names) is list else names
        t_names = target_name[idx] if type(target_name) is list else target_name
        output = images_s_style.detach()[idx]
        output = torch.clip(output, 0, 1)
        output = output * 255.
        output = torch.moveaxis(output, 0, -1)
        if time_iter == 1:
            image_name = 'warmup/{}_iter{:d}_{}_2_{}.jpg'.format(args.mode, i_iter + 1,
                                                                 Path(s_name).stem,
                                                                 Path(t_names).stem)
        else:
            if not os.path.exists('{}/{}'.format(args.style_dir, apdx)):
                os.mkdir('{}/{}'.format(args.style_dir, apdx))
            image_name = '{}/{}_iter{:d}_{}_2_{}_{}.jpg'.format(apdx, args.mode, i_iter + 1,
                                                                Path(s_name).stem,
                                                                Path(t_names).stem, i)

            image_name = '{}/{}'.format(args.style_dir, image_name)
        cv2.imwrite(image_name, output.cpu().numpy().astype(np.uint8))
        print("{} saved.".format(image_name))


@timer.timeit
def save_transferred_images_RAIN(images_s_style, c_names, s_names, epoch, iter='', idx_to_save=(0,),
                                 save_dir='RAIN_out', stage='pretrain', normalization='minmax'):
    for idx in idx_to_save:
        c_name = c_names[idx] if (type(c_names) is list or type(c_names) is tuple) else c_names
        s_name = s_names[idx] if (type(s_names) is list or type(s_names) is tuple) else s_names
        output = images_s_style.detach()[idx]
        if normalization == 'zscore':
            output = (output - output.min()) / (output.max() - output.min())
        elif normalization == 'minmax':
            output = torch.clip(output, 0, 1)
        else:
            raise NotImplementedError
        output = output * 255.
        output = torch.moveaxis(output, 0, -1)
        image_name = os.path.join(save_dir, '{}/e{:d}_{}_2_{}_{}.png'.format(stage, epoch + 1,
                                                                             Path(c_name).stem,
                                                                             Path(s_name).stem, iter), )
        if not Path(image_name).parent.exists():
            Path(image_name).parent.mkdir(parents=True)
        cv2.imwrite(image_name, output.cpu().numpy().astype(np.uint8))
        print("{} saved.".format(image_name))


def crop_normalize(img_style, img_s, normalization):
    style_size = img_style.size()
    img_size = img_s.size()
    if style_size[-1] != img_size[-1] or style_size[-2] != img_size[-2]:
        img_style = center_crop(img_style, [img_size[-2], img_size[-1]])
    # if normalization == 'zscore':
    #     img_style = (img_style - img_style.mean(dim=(1, 2, 3), keepdim=True)) / img_style.std(dim=(1, 2, 3), keepdim=True)
    return img_style


def get_pretrained_checkpoint(backbone):
    # --- UPDATE THIS PATH to your Kaggle dataset containing the .pth file ---
    # Example: PRETRAINED_BASE_PATH = Path("/kaggle/input/my-resnet-weights/")
    PRETRAINED_BASE_PATH = Path("/kaggle/input/resnet50-pytorch-pretrained/") # Make sure this dataset is added to your notebook
    
    WEIGHT_PATHS = {
        'resnet50': PRETRAINED_BASE_PATH / 'resnet50-19c8e357.pth', # Ensure this filename matches your uploaded file
        # Add other backbones if you have their weights locally
    }
    checkpoint_path = WEIGHT_PATHS.get(backbone)

    if checkpoint_path is None:
        # This case should ideally not be hit if Trainer_baseline.prepare_model calls this only for known backbones.
        # If smp.Unet handles 'imagenet' download, this function's role is for *local* fallbacks or specific weights.
        print(f"Warning: Pretrained weights for backbone '{backbone}' are not locally defined in WEIGHT_PATHS. Relaying on smp.Unet 'imagenet' download.")
        return None # Allow smp.Unet to try downloading
    
    print(f"Attempting to load LOCAL pretrained weights for {backbone} from: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        print(f"ERROR: LOCAL Pretrained weights file not found at {checkpoint_path}!")
        print("Ensure the Kaggle input dataset is correctly added and the path/filename is exact.")
        print("Will rely on smp.Unet to download 'imagenet' weights if possible.")
        return None # Allow smp.Unet to try downloading if local file is explicitly configured but not found
        
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) 
    print(f"--- Successfully loaded LOCAL pretrained checkpoint for {backbone} from {checkpoint_path}. ---")
    return checkpoint


def save_nii(data, affine, file_name):
    img = nib.Nifti1Image(data, affine)
    img.to_filename(file_name)


def name_the_model(model_name, model_dir, apdx='', data=None):
    if model_name is None:
        model_name = ''
        pattern = re.compile("pat_?\d+")
        m = re.search(pattern, model_dir)
        if m is not None:
            model_name += m.group() + '.'
        if "best_unet_model_checkpoint_point.lr0.0002.oneshot.adam.pat10.slc13.dr0.01.offdecay.t2.e25.Scr0.411.pt" in model_dir:
            model_name += 'baseline'
        else:
            if data is None:
                if 'one' in model_dir:
                    model_name += 'one'
                elif 'few' in model_dir:
                    model_name += 'few'
                else:
                    model_name += 'full'
            else:
                model_name += data
            if 'Stride' in model_dir:
                model_name += '.Stride'
            if 'Block' in model_dir:
                model_name += '.Block'
            if 'mse' in model_dir:
                model_name += '.CNR'
            if 'd1' in model_dir:
                model_name += '.d1'
            if 'd2' in model_dir:
                model_name += '.d2'
            if 'clda' in model_dir:
                model_name += '.C'
            if 'eps' in model_dir:
                model_name += '.S'
            if 'nopre' in model_dir:
                model_name += '.nopre'
            m = re.search('\.s\d+\.', model_dir)
            if m is not None:
                split = int(m.group()[2:-1])
            else:
                split = 0
            model_name += f'.s{split}'
            model_name += '.f1' if '.f1.' in model_dir else '.f0'
            if 'Base' in model_dir:
                model_name += '.Base'
                if 'trainWt' in model_dir:
                    model_name += '.withT'
            if 'AdaptSeg' in model_dir:
                model_name += '.AdaptSeg'
            if 'Advent' in model_dir:
                model_name += '.Advent'
            if 'AdaptEvery' in model_dir:
                model_name += '.AdaptEvery'
            if 'SIFA' in model_dir:
                model_name += '.SIFA'
            if 'BCL' in model_dir:
                model_name += '.BCL'
            if 'DDFSeg' in model_dir:
                model_name += '.DDFSeg'
            if 'MPSCL' in model_dir:
                model_name += '.MPSCL'
            if 'MCCL' in model_dir:
                model_name += '.MCCL'
                if 'norain' in model_dir:
                    model_name += '.norain'
                if 'CNR' in model_dir:
                    model_name += '.CNR'
                if 'w_ave' not in model_dir:
                    model_name += '.noWav'
                if '.p1.' in model_dir:
                    model_name += '.p1'
                    m = re.search(re.compile("thd\d\.\d"), model_dir)
                if m is not None:
                    model_name += f".{m.group()}"
            m = re.search('p\d+\.', model_dir)
            if m is not None:
                model_name += f'.{m.group()[:-1]}'
        if apdx != '':
            model_name += '.{}'.format(apdx)
    return model_name


def write_model_graph(model, images, log_dir='../runs/model_graph'):
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    import os
    from pathlib import Path
    if os.path.exists(log_dir):
        now = datetime.now()
        log_dir += f'.{now.hour}.{now.minute}'
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_graph(model, images)
    writer.close()
    print(f'graph saved at: {Path(writer.log_dir).absolute()}')


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def plot_img_preds(imgs: torch.Tensor, preds: torch.Tensor, imgs_style: torch.Tensor, preds_style: torch.Tensor,
                   cmap='gray'):
    from matplotlib import pyplot as plt
    if imgs.dim() == 4:
        temp = imgs[0, 0]
    elif imgs.dim() == 3:
        temp = imgs[0]
    elif imgs.dim() == 2:
        temp = imgs
    plt.imshow(temp.detach().cpu().numpy(), cmap=cmap)
    plt.show()
    if preds.dim() == 4:
        temp = torch.argmax(preds, dim=1)[0]
    if preds.dim() == 3:
        temp = torch.argmax(preds, dim=0)
    if preds.dim() == 2:
        temp = preds
    plt.imshow(temp.detach().cpu().numpy(), cmap=cmap)
    plt.show()
    if imgs_style.dim() == 4:
        temp = imgs_style[0, 0]
    elif imgs_style.dim() == 3:
        temp = imgs_style[0]
    elif imgs_style.dim() == 2:
        temp = imgs_style
    plt.imshow(temp.detach().cpu().numpy(), cmap=cmap)
    plt.show()
    if preds_style.dim() == 4:
        temp = torch.argmax(preds_style, dim=1)[0]
    if preds_style.dim() == 3:
        temp = torch.argmax(preds_style, dim=0)
    if preds_style.dim() == 2:
        temp = preds_style
    plt.imshow(temp.detach().cpu().numpy(), cmap=cmap)
    plt.show()


# offset = 1
# plot_img_preds(images_s[offset], pred[pred_s_size // 2 + offset], images_s_style[offset], pred[offset])
# plot_img_preds(images_t_temp[offset], pseudo_label_t_temp[offset], img_t_aug_temp[offset], pseudo_label_t_temp[t_size + offset])
# mean, std = torch.mean(torch.max(pseudo_label_t_temp[t_size + offset], 0).values).detach().cpu().numpy(), torch.std(torch.max(pseudo_label_t_temp[t_size + offset], 0).values).detach().cpu().numpy()
# plt.imshow((torch.argmax(pseudo_label_t_temp[t_size + offset], 0) * torch.where(torch.max(pseudo_label_t_temp[t_size + offset], 0).values > float(mean - 0.7 * std), 1, 0)).detach().cpu().numpy(), cmap='gray')
# mean, std = torch.mean(torch.max(pseudo_label_t_temp[offset], 0).values).detach().cpu().numpy(), torch.std(torch.max(pseudo_label_t_temp[offset], 0).values).detach().cpu().numpy()
# plt.imshow((torch.argmax(pseudo_label_t_temp[offset], 0) * torch.where(torch.max(pseudo_label_t_temp[offset], 0).values > float(mean - 0.7 * std), 1, 0)).detach().cpu().numpy(), cmap='gray')
# print(torch.min(pred).detach().cpu().numpy(), torch.max(pred).detach().cpu().numpy())


def plot_tool_noaxis(img, cmap=None):
    from matplotlib import pyplot as plt
    plt.axis('off')
    plt.tight_layout()
    if cmap is None:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
    plt.show()


def find_class_prior_mscmrseg():
    bg, myo, lv, rv = 0, 0, 0, 0
    n_img = 0
    for file in glob('../../../data/mscmrseg/origin/trainBmask/*lge*.png'):
        mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        bg += np.count_nonzero(mask == 0)
        myo += np.count_nonzero(mask == 85)
        lv += np.count_nonzero(mask == 212)
        rv += np.count_nonzero(mask == 255)
        n_img += 1
    print(f"number of images: {n_img}")
    print(f"total number of pixel for each class: bg {bg}, myo {myo}, lv {lv}, rv {rv}")
    print(f"In average, each image has bg {bg / n_img}, myo {myo / n_img}, lv {lv / n_img}, rv {rv / n_img}")
    total = bg + myo + lv + rv
    print(
        f"Density of pixel for each class: bg {bg / total:.4f}, myo {myo / total:.4f}, lv {lv / total:.4f}, rv {rv / total:.4f}")


def generate_train_test_split():
    import numpy as np
    all = np.arange(1, 21)
    train = np.sort(np.random.choice(all, 10, replace=False))
    test = np.setdiff1d(all, train)
    train_text = ""
    for i in train:
        train_text += f'{i}, '
    train_text = train_text[:-2]
    test_text = ""
    for i in test:
        test_text += f'{i}, '
    test_text = test_text[:-2]
    print(train_text)
    print(test_text)


def check_mask_with_label():
    for i in range(1001, 1021):
        mask = nib.load(f'F:\data\mmwhs/affregcommon2mm_roi_mr_train/roi_mr_train_{i}_label.nii').get_fdata()
        mask = (mask == 205) * 1 + (mask == 500) * 2 + (mask == 600) * 3
        tmp1 = np.unique(np.where(mask == 1)[1])
        tmp2 = np.unique(np.where(mask == 2)[1])
        tmp3 = np.unique(np.where(mask == 3)[1])
        tmp = np.intersect1d(np.intersect1d(tmp1, tmp2), tmp3)
        print(f'patient {i}, len: {len(tmp)}, idx: {tmp}')


def load_raw_data_mmwhs(img_path, mask_path=None):
    print(f"DEBUG: Worker attempting to read file: {img_path}")

    try:
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    except RuntimeError as e:
        # Bắt lỗi SimpleITK để in ra thông tin chi tiết hơn nếu cần
        print(f"ERROR: SimpleITK failed to read {img_path}. Error: {e}")
        raise
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    img = np.pad(img[:, 8: -8, 0], ((2, 2), (0, 0)), constant_values=img.min())
    if mask_path is not None:
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        mask = np.pad(mask[:, 8: -8, 0], ((2, 2), (0, 0)))
        mask = (mask == 205) * 1 + (mask == 500) * 2 + (mask == 600) * 3
        mask = np.array(mask, dtype=np.uint8)
    else:
        mask = None
    return img, mask


def find_backbone(model_dir):
    if 'DDFSeg' in model_dir:
        return 'DDFSeg'
    elif 'resnet50' in model_dir:
        return 'resnet50'
    elif 'se_resnet50' in model_dir:
        return 'se_resnet50'
    elif 'se_resnet101' in model_dir:
        return 'se_resnet101'
    elif 'efficientnet-b5' in model_dir:
        return 'efficientnet-b5'
    elif 'efficientnet-b6' in model_dir:
        return 'efficientnet-b6'
    elif 'mobilenet' in model_dir:
        return 'mobilenet'
    elif 'inceptionv4' in model_dir:
        return 'inceptionv4'
    elif 'xceptionv4' in model_dir:
        return 'xceptionv4'
    elif 'densenet161' in model_dir:
        return 'densenet161'
    else:
        raise NotImplementedError


def load_model(model_dir, args=None):
    from model.DRUNet import Segmentation_model as DR_UNet
    from model.deeplabv2 import get_deeplab_v2
    backbone = find_backbone(model_dir)
    if 'drunet' in model_dir:
        segmentor = DR_UNet(n_class=4, multilvl=True if 'mutlvl' in model_dir else False)
    elif 'deeplab' in model_dir:
        segmentor = get_deeplab_v2(num_classes=4, multi_level=True if 'mutlvl' in model_dir else False,
                                   input_size=224)
    else:
        if backbone == 'DDFSeg':
            from model.DDFSeg import DDFNet, SegDecoder
            from torch.nn import Sequential
            segdecoder = SegDecoder()
            ddfnet = DDFNet()
            segmentor = Sequential(ddfnet.encoderc, ddfnet.encodert, segdecoder)
        else:
            if 'AdaptEvery' in model_dir:
                from model.segmentation_models import segmentation_model_point
                segmentor = segmentation_model_point(name=backbone, pretrained=False,
                                                     decoder_channels=(512, 256, 128, 64, 32), in_channel=3,
                                                     classes=4, multilvl=True, fc_inch=4, extpn=False)
            else:
                from model.segmentation_models import segmentation_models
                segmentor = segmentation_models(name=backbone, pretrained=False,
                                                decoder_channels=(512, 256, 128, 64, 32), in_channel=3,
                                                classes=4, multilvl=True if ('mutlvl' in model_dir)
                                                                            or ('MPSCL' in model_dir) else False,
                                                args=args)
    return segmentor


def load_mnmx_csv(modality, percent=99):
    base_csv_path = Path("/kaggle/input/generate-normalization-stats/normalization_stats") # Your Kaggle Dataset path
    percent_for_filename = int(float(percent))
    csv_filename = f"{modality.upper()}minmax{percent_for_filename}.csv"
    csv_file = base_csv_path / csv_filename
    print(f"Attempting to load minmax CSV for modality '{modality}' with percent '{percent_for_filename}' from: {csv_file}")
    if not csv_file.exists():
        raise FileNotFoundError(f"Minmax CSV file not found: {csv_file}.")
    mnmx = pd.read_csv(csv_file, index_col=0)
    expected_min_col = f'min{percent_for_filename}'
    expected_max_col = f'max{percent_for_filename}'
    if not (expected_min_col in mnmx.columns and expected_max_col in mnmx.columns):
        # Fallback for "min99" if using percent=99
        if percent_for_filename == 99 and 'min99' in mnmx.columns and 'max99' in mnmx.columns:
            print(f"Note: Using 'min99'/'max99' columns from {csv_file} as '{expected_min_col}'/'{expected_max_col}' not found.")
        else:
            raise ValueError(f"CSV file {csv_file} does not contain expected columns '{expected_min_col}' and '{expected_max_col}'. Found: {mnmx.columns.tolist()}")
    return mnmx


def assert_match(img_path, lab_path):
    img_sample = re.search('img\d+_', str(Path(img_path).stem)).group()[3:-1]
    lab_sample = re.search('lab\d+_', str(Path(lab_path).stem)).group()[3:-1]
    assert img_sample == lab_sample, f"The img sample number does not match with the lab sample number; img: {img_sample}, lab: {lab_sample}"


def save_batch_data(data_dir, img, lab, name, normalization='minmax', aug_mode='simple'):
    if img.ndim == 4:
        for im, gt, nm in zip(img, lab, name):
            save_batch_data(data_dir, im[1], gt, nm, normalization, aug_mode)
    elif img.ndim == 2:
        if 'minmax' in normalization:
            img = np.array(img * 255, dtype=np.uint8)
        else:
            img = np.array(255 * ((img - img.min()) / (img.max() - img.min())), np.uint8)
        img_path = str(Path(data_dir).parent.joinpath(f'batch_data_{aug_mode}').joinpath(f'{name}.png'))
        if not Path(img_path).parent.exists():
            Path(img_path).parent.mkdir(parents=True)
            print(f'{Path(img_path).parent} created.')
        if not Path(img_path).exists():
            mask = (lab == 1) * 87 + (lab == 2) * 212 + (lab == 3) * 255
            img = np.concatenate([img, mask], axis=1)
            cv2.imwrite(img_path, img)
            # lab_path = str(Path(data_dir).parent.joinpath(f'batch_data_{aug_mode}').joinpath(f'{name}.png'.replace('img', 'lab')))
            # cv2.imwrite(lab_path, mask)
    else:
        raise NotImplementedError


def check_mkdir_parent_dir(path):
    if not Path(path).parent.exists():
        Path(path).parent.mkdir(parents=True)


def check_del(path):
    path_dir = Path(path)
    if Path(path).exists():
        if path_dir.is_dir():
            import shutil
            shutil.rmtree(path_dir)
        else:
            os.remove(path_dir)


def easy_dic(dic):
    dic = EasyDict(dic)
    for key, value in dic.items():
        if isinstance(value, dict):
            dic[key] = EasyDict(value)
    return dic


def convert_plain_dict(dic):
    new_dict = {}
    for key, value in dic.items():
        if type(value) == dict:
            new_dict.update(convert_plain_dict(value))
        else:
            new_dict[key] = value
    return new_dict


def mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)
        print('DIR {} created'.format(p))
    return p


def show_config(cfg):
    msg = ''
    for key, value in cfg.items():
        if isinstance(value, dict):
            continue
        else:
            msg += '{:>25} : {:<15}\n'.format(key, value)
    return msg


def thres_cb_plabel(P, thres_dic, num_cls=19, label=None):
    """
    Generate global label, so called in the paper. The mask of the prediction where the probability is greater than
    the predefined threshold for each category.
    :param P: the prediction
    :param thres_dic: the threshold
    :param num_cls: number of classes
    :param label: the (pseudo)label
    :return: the mask, and the coordinate of the masked pixels
    """
    P = F.softmax(P, dim=1)
    n, c, h, w = P.shape  # 1, c, h, w
    if label is not None:
        pred_label = label
    else:
        pred_label = P.argmax(dim=1).squeeze()  # take the prediction as the (pseudo)label
    # import pdb; pdb.set_trace()
    vec = [torch.Tensor([thres_dic[k]]) for k in range(num_cls)]
    vec = torch.stack(vec).cuda()
    vec = vec.view(1, c, 1, 1)  # the threshold value for each class
    vec = vec.expand_as(P)  # (1, c, h, w)
    # mask = torch.gt(P, vec)
    mask = torch.ge(P, vec)  # mask out the predictions with its probability greater than the threshold
    # the pixel with probability smaller than the threshold is set to 0, otherwise >= 1. But it is possible that the
    # value is greater than 1
    mask = torch.clip(mask.sum(dim=1), 0, 1).byte()  # (1, h, w)

    # plabel = torch.where(mask, pred_label, 255)
    plabel = torch.where(P > vec, )  # the coordinate where P > vec
    mask = mask.squeeze()  # (h, w)
    plabel = plabel.squeeze()

    return mask, plabel


def gene_plabel_prop(P, prop):
    """
    Generate local label, so called in the paper. The mask out of the prediction where the probability is greater than
    the predefined threshold, and the pseudo labels according to the mask. Set pixels to 255 if no selected by the mask.
    :param P: the prediction (1, c, h, w)
    :param prop: default = .1.
    :return:
    """
    assert P.dim() == 4
    _, C, H, W = P.shape
    total_cnt = float(H * W)  # total number of pixels

    P = F.softmax(P, dim=1)
    pred_label = P.argmax(dim=1).squeeze()  # (h, w) predicted class index
    pred_prob = (P.max(dim=1)[0]).squeeze()  # (h, w) the max probability values in the prediction
    plabel = torch.ones_like(pred_label, dtype=torch.long)
    plabel = plabel * 255
    thres_index = int(prop * total_cnt) - 1
    # sort the max prob of each pixel from the greatest to the smallest
    value, _ = torch.sort(pred_prob.view(1, -1), descending=True)
    thres = value[0, thres_index]  # find the threshold that extracts the top :prop ratio of prob in the predictions
    select_mask = (pred_prob > thres).cuda()  # (h, w)
    plabel = torch.where(select_mask, pred_label, plabel)
    return select_mask, plabel


def mask_fusion(P, mask1, mask2, label=None):
    """
    fuse the global and local mask and generate the pseudo label with the mask
    :param P: (1, c, h, w)
    :param mask1: (h, w)
    :param mask2: (h, w)
    :param label: the (pseudo)label
    :return: mask (h, w), plabel(h, w)
    """
    assert mask1.shape == mask2.shape
    mask = mask1 + mask2  # either global mask, or local mask
    _, C, H, W = P.shape
    if label is None:
        P = F.softmax(P, dim=1)
        pred_label = P.argmax(dim=1).squeeze()  # (h, w)
    else:
        pred_label = label.squeeze()  # (h, w) expected
    plabel = torch.where(mask > 0, pred_label, 255)
    return mask, plabel


def Acc(plabel, label, num_cls=19):
    """
    calculate the accuracies and proportion from the pseudo labels and the true labels
    :param plabel: (h, w) the pseudo label generated from the prediction
    :param label: (1, h, w) the true label
    :param num_cls:
    :return:    acc: the accuracy for the overall predictions that is over the threshold. sensitivity: TP / TP + FN
                prop: the proportion of the selected pixels whose prediction is larger than the threshold.
                acc_dict: {class_index: (accuracy, total number of prediction of the class)}
    """
    plabel = plabel.view(-1, 1).squeeze().cuda().float()
    label = label.view(-1, 1).squeeze().cuda().float()
    total = label.shape[0]  # the total number of pixels
    mask = (plabel != 255) * (label != 255)

    label = torch.masked_select(label, mask)
    plabel = torch.masked_select(plabel, mask)
    correct = label == plabel  # the correct mask where the correct pixel is 1, others is 0
    acc_dict = {}  # {class_index: (accuracy, total number of prediction of the class)}
    for i in range(num_cls):
        cls_mask = plabel == i
        cls_total = cls_mask.sum().float().item()  # the total number of pixel assigned to class i in pseudo label
        if cls_total == 0:
            continue  # WARNING: some of the class indices in :acc_dict may be missing.
        cls_correct = (cls_mask * correct).sum().float().item()  # the number of corrected classified pixels for class i
        cls_acc = cls_correct / cls_total  # sensitivity: TP / TP + FN
        acc_dict[i] = (cls_acc, cls_total)
    correct_cnt = correct.sum().float()  # the total number of correct predictions
    selected = mask.sum().float()  # the total number of selected predictions
    if selected == 0:
        acc = 0.0
    else:
        # the accuracy for the overall predictions that is over the threshold. sensitivity: TP / TP + FN
        acc = correct_cnt / selected
    prop = selected / total  # the proportion of the selected pixels whose prediction is larger than the threshold.
    return acc, prop, acc_dict


if __name__ == '__main__':
    print('utils_.py finished its main block execution (if any).')