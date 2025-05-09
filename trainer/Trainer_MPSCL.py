# %%writefile /kaggle/working/Soft-Labeled-Contrastive-Learning/trainer/Trainer_MPSCL.py
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import sys
from pathlib import Path # Ensure Path is imported

"""torch import"""
import torch
import torch.nn.functional as F

"""utils import"""
from utils.loss import loss_calc, MPCL, dice_loss, mpcl_loss_calc
from utils.utils_ import update_class_center_iter, generate_pseudo_label, prob_2_entropy

"""evaluator import"""
from evaluator import Evaluator

"""trainer import"""
from trainer.Trainer_Advent import Trainer_Advent
import config # Import config to access NUM_CLASSES if needed elsewhere

class Trainer_MPSCL(Trainer_Advent):
    def __init__(self):
        super().__init__()

    def add_additional_arguments(self):
        super(Trainer_MPSCL, self).add_additional_arguments()
        self.parser.add_argument('-adjust_lr', action='store_true')

        # Using defaults set by user's previous direct edits
        self.parser.add_argument('-src_temp', type=float, default=0.1)
        self.parser.add_argument('-src_base_temp', type=float, default=1)
        self.parser.add_argument('-trg_temp', type=float, default=0.1)
        self.parser.add_argument('-trg_base_temp', type=float, default=1)
        self.parser.add_argument('-src_margin', type=float, default=.4)
        self.parser.add_argument('-trg_margin', type=float, default=.2)

        self.parser.add_argument('-class_center_m', type=float, default=0.9)
        self.parser.add_argument('-pixel_sel_th', type=float, default=.25)

        self.parser.add_argument('-w_mpcl_s', type=float, default=1.0)
        self.parser.add_argument('-w_mpcl_t', type=float, default=1.0)

        self.parser.add_argument('-dis_type', type=str, default='origin')

        # Add definitions for -part and -CNR_w if not present, using safe defaults
        # Check if action already exists before adding
        if '-part' not in self.parser._option_string_actions:
             self.parser.add_argument('-part', type=int, default=1,
                                      help='number of partitions for rMC (set to 2 via cmd/hardcode)')
        if '-CNR_w' not in self.parser._option_string_actions:
             self.parser.add_argument('-CNR_w', type=float, default=0.0,
                                      help='Weight for CNR loss (set via cmd/hardcode)')

    def get_arguments_apdx(self):
        super(Trainer_MPSCL, self).get_basic_arguments_apdx(name='MPSCL')
        if self.args.multilvl:
            pass # Keep name simpler as aux loss is disabled
        self.apdx += f".bs{self.args.bs}"
        self.apdx += f".lr_dis{self.args.lr_dis}.w_dis{self.args.w_dis}"
        if self.args.multilvl:
            self.apdx += f'.w_d_aux{self.args.w_dis_aux}'
        self.apdx += f'.w_mpscl_s{self.args.w_mpcl_s}.t{self.args.w_mpcl_t}'
        part_num = getattr(self.args, 'part', 1)
        cnr_weight = getattr(self.args, 'CNR_w', 0.0)
        if part_num > 1:
             self.apdx += f'.p{part_num}'
        if cnr_weight > 0:
             self.apdx += f'.cnr{cnr_weight}'

    def prepare_losses(self):
        self.mpcl_loss_src = MPCL(self.device, num_class=self.args.num_classes, temperature=self.args.src_temp,
                                  base_temperature=self.args.src_base_temp, m=self.args.src_margin)

        self.mpcl_loss_trg = MPCL(self.device, num_class=self.args.num_classes, temperature=self.args.trg_temp,
                                  base_temperature=self.args.trg_base_temp, m=self.args.trg_margin)
        self.mse_loss = torch.nn.MSELoss()


    def train_epoch(self, epoch):
        # --- (train_epoch code as corrected previously - removing 'partition' arg) ---
        # print(f'start to train epoch: {epoch}') # Print is inside train loop now
        self.segmentor.train()
        self.d_main.train()
        if self.args.multilvl:
             self.d_aux.train()

        results = {}
        source_domain_label = 1
        target_domain_label = 0
        loss_seg_list, loss_uncertainty, loss_prior_list = [], [], []
        loss_seg_aux_list = []
        loss_adv_list, loss_adv_aux_list, loss_dis_list, loss_dis_aux_list = [], [], [], []
        loss_mpcl_tr_list, loss_mpcl_tg_list = [], []
        loss_cnr_list = []
        d_acc_s, d_acc_t = [], []
        d_aux_acc_s, d_aux_acc_t = [], []

        partition_num = getattr(self.args, 'part', 1)
        cnr_weight = getattr(self.args, 'CNR_w', 0.0)

        for batch_content, batch_style in zip(self.content_loader, self.style_loader):
            self.opt_d.zero_grad()
            self.opt.zero_grad()
            for param in self.d_main.parameters():
                param.requires_grad = False
            if self.args.multilvl:
                self.opt_d_aux.zero_grad()
                for param in self.d_aux.parameters():
                    param.requires_grad = False

            img_s, labels_s, names = batch_content
            img_s, labels_s = img_s.to(self.device, non_blocking=self.args.pin_memory), \
                              labels_s.to(self.device, non_blocking=self.args.pin_memory)
            img_t, labels_t, namet = batch_style
            img_t = img_t.to(self.device, non_blocking=self.args.pin_memory)

            out_s = self.segmentor(img_s, features_out=True)
            pred_s_main, pred_s_aux, dcdr_ft_s = out_s
            out_t = self.segmentor(img_t, features_out=True)
            pred_t_main, pred_t_aux, dcdr_ft_t = out_t

            loss_seg = loss_calc(pred_s_main, labels_s, self.device, False) + dice_loss(pred_s_main, labels_s)
            loss_seg_list.append(loss_seg.item())

            # Aux loss disabled
            total_loss_s = loss_seg
            loss_seg_aux = torch.tensor(0.0, device=self.device)
            loss_seg_aux_list.append(loss_seg_aux.item())

            self.centroid_s = update_class_center_iter(dcdr_ft_s, labels_s, self.centroid_s,
                                                       m=self.args.class_center_m)
            hard_pixel_label, pixel_mask = generate_pseudo_label(dcdr_ft_t, self.centroid_s, self.args.pixel_sel_th)

            # MPCL Source Loss (partition removed)
            mpcl_loss_tr = mpcl_loss_calc(feas=dcdr_ft_s, labels=labels_s,
                                          class_center_feas=self.centroid_s.detach(),
                                          loss_func=self.mpcl_loss_src, tag='source')
            loss_mpcl_tr_list.append(mpcl_loss_tr.item())

            # MPCL Target Loss (partition removed)
            mpcl_loss_tg = mpcl_loss_calc(feas=dcdr_ft_t, labels=hard_pixel_label,
                                          class_center_feas=self.centroid_s.detach(),
                                          loss_func=self.mpcl_loss_trg,
                                          pixel_sel_loc=pixel_mask, tag='target')
            loss_mpcl_tg_list.append(mpcl_loss_tg.item())

            # CNR Loss (Placeholder logic)
            loss_cnr = torch.tensor(0.0, device=self.device)
            if cnr_weight > 0:
                try:
                    from utils.utils_ import cal_centroid
                    centroid_t = cal_centroid(features=dcdr_ft_t, pseudo_label=hard_pixel_label,
                                              num_cls=self.args.num_classes, pixel_mask=pixel_mask)
                    centroid_s_norm = torch.norm(self.centroid_s.detach(), p=2, dim=1)
                    centroid_t_norm = torch.norm(centroid_t, p=2, dim=1)
                    loss_cnr = self.mse_loss(centroid_s_norm, centroid_t_norm)
                    loss_cnr_list.append(loss_cnr.item())
                except ImportError:
                    print("WARNING: utils.utils_.cal_centroid not found. Cannot calculate CNR loss.")
                    loss_cnr_list.append(loss_cnr.item())
                except Exception as e:
                    print(f"WARNING: Error during CNR calculation: {e}. Skipping CNR loss.")
                    loss_cnr_list.append(loss_cnr.item())
            else:
                loss_cnr_list.append(loss_cnr.item())

            # Adversarial Loss
            pred_t_softmax = F.softmax(pred_t_main, dim=1)
            uncertainty_mapT = prob_2_entropy(pred_t_softmax)
            D_out_main = self.d_main(uncertainty_mapT)
            loss_adv_main = F.binary_cross_entropy_with_logits(D_out_main, torch.FloatTensor(
                D_out_main.data.size()).fill_(source_domain_label).to(self.device))
            loss_adv_list.append(loss_adv_main.item())
            loss_adv = self.args.w_dis * loss_adv_main

            loss_adv_aux = torch.tensor(0.0, device=self.device)
            if self.args.multilvl:
                try:
                    pred_t_softmax_aux = F.softmax(pred_t_aux, dim=1)
                    uncertainty_mapT_aux = prob_2_entropy(pred_t_softmax_aux)
                    D_out_aux = self.d_aux(uncertainty_mapT_aux)
                    loss_adv_aux = F.binary_cross_entropy_with_logits(D_out_aux, torch.FloatTensor(
                        D_out_aux.data.size()).fill_(source_domain_label).to(self.device))
                    loss_adv += self.args.w_dis_aux * loss_adv_aux
                except Exception as e:
                     print(f"Warning: Error calculating aux adv loss: {e}. Skipping.")
                     loss_adv_aux = torch.tensor(0.0, device=self.device)
            loss_adv_aux_list.append(loss_adv_aux.item())

            # Total Generator Loss
            total_generator_loss = (total_loss_s
                                   + loss_adv
                                   + self.args.w_mpcl_s * mpcl_loss_tr
                                   + self.args.w_mpcl_t * mpcl_loss_tg
                                   + cnr_weight * loss_cnr)
            total_generator_loss.backward()

            # Train Discriminators
            for param in self.d_main.parameters():
                param.requires_grad = True
            if self.args.multilvl:
                for param in self.d_aux.parameters():
                    param.requires_grad = True

            # Source
            pred_s_main_d = pred_s_main.detach()
            d_out_main_s = self.d_main(prob_2_entropy(F.softmax(pred_s_main_d, dim=1)))
            loss_d_main_s = F.binary_cross_entropy_with_logits(d_out_main_s, torch.FloatTensor(
                d_out_main_s.data.size()).fill_(source_domain_label).to(self.device))
            loss_d_main_s = loss_d_main_s / 2.0
            loss_d_main_s.backward()
            D_out_s_main = torch.sigmoid(d_out_main_s.detach()).cpu().numpy()
            D_out_s_main = np.where(D_out_s_main >= .5, 1, 0)
            d_acc_s.append(np.mean(D_out_s_main))

            loss_d_aux_s_val = 0.0
            if self.args.multilvl:
                try:
                    pred_s_aux_d = pred_s_aux.detach()
                    d_out_aux_s = self.d_aux(prob_2_entropy(F.softmax(pred_s_aux_d, dim=1)))
                    loss_d_aux_s = F.binary_cross_entropy_with_logits(d_out_aux_s, torch.FloatTensor(
                        d_out_aux_s.data.size()).fill_(source_domain_label).to(self.device))
                    loss_d_aux_s = loss_d_aux_s / 2.0
                    loss_d_aux_s.backward()
                    loss_d_aux_s_val = loss_d_aux_s.item()
                    D_out_s_aux = torch.sigmoid(d_out_aux_s.detach()).cpu().numpy()
                    D_out_s_aux = np.where(D_out_s_aux >= .5, 1, 0)
                    d_aux_acc_s.append(np.mean(D_out_s_aux))
                except Exception as e:
                     print(f"Warning: Error calculating aux dis source loss: {e}. Skipping.")
                     d_aux_acc_s.append(0.0)
            else:
                d_aux_acc_s.append(0.0)

            # Target
            pred_t_main_d = pred_t_main.detach()
            d_out_main_t = self.d_main(prob_2_entropy(F.softmax(pred_t_main_d, dim=1)))
            loss_d_main_t = F.binary_cross_entropy_with_logits(d_out_main_t, torch.FloatTensor(
                d_out_main_t.data.size()).fill_(target_domain_label).to(self.device))
            loss_d_main_t = loss_d_main_t / 2.0
            loss_d_main_t.backward()
            loss_dis_list.append(loss_d_main_s.item() + loss_d_main_t.item())
            D_out_t_main = torch.sigmoid(d_out_main_t.detach()).cpu().numpy()
            D_out_t_main = np.where(D_out_t_main >= .5, 1, 0)
            d_acc_t.append(1 - np.mean(D_out_t_main))

            loss_d_aux_t_val = 0.0
            if self.args.multilvl:
                try:
                    pred_t_aux_d = pred_t_aux.detach()
                    d_out_aux_t = self.d_aux(prob_2_entropy(F.softmax(pred_t_aux_d, dim=1)))
                    loss_d_aux_t = F.binary_cross_entropy_with_logits(d_out_aux_t, torch.FloatTensor(
                        d_out_aux_t.data.size()).fill_(target_domain_label).to(self.device))
                    loss_d_aux_t = loss_d_aux_t / 2.0
                    loss_d_aux_t.backward()
                    loss_d_aux_t_val = loss_d_aux_t.item()
                    D_out_t_aux = torch.sigmoid(d_out_aux_t.detach()).cpu().numpy()
                    D_out_t_aux = np.where(D_out_t_aux >= .5, 1, 0)
                    d_aux_acc_t.append(1 - np.mean(D_out_t_aux))
                except Exception as e:
                     print(f"Warning: Error calculating aux dis target loss: {e}. Skipping.")
                     d_aux_acc_t.append(0.0)
            else:
                d_aux_acc_t.append(0.0)

            if self.args.multilvl:
                 loss_dis_aux_list.append(loss_d_aux_s_val + loss_d_aux_t_val)
            else:
                 loss_dis_aux_list.append(0.0)

            # Optimizer Steps
            self.opt.step()
            self.opt_d.step()
            if self.args.multilvl:
                self.opt_d_aux.step()

        # Aggregate results
        results['seg_s'] = np.mean(loss_seg_list) if loss_seg_list else 0
        results['dis_acc_s'] = np.mean(d_acc_s) if d_acc_s else 0
        results['dis_acc_t'] = np.mean(d_acc_t) if d_acc_t else 0
        results['loss_adv'] = np.mean(loss_adv_list) if loss_adv_list else 0
        results['loss_dis'] = np.mean(loss_dis_list) if loss_dis_list else 0
        results['loss_mpscl_tr'] = np.mean(loss_mpcl_tr_list) if loss_mpcl_tr_list else 0
        results['loss_mpscl_tg'] = np.mean(loss_mpcl_tg_list) if loss_mpcl_tg_list else 0
        results['loss_cnr'] = np.mean(loss_cnr_list) if loss_cnr_list else 0
        results['seg_s_aux'] = np.mean(loss_seg_aux_list) if loss_seg_aux_list else 0
        results['loss_adv_aux'] = np.mean(loss_adv_aux_list) if loss_adv_aux_list else 0
        results['loss_dis_aux'] = np.mean(loss_dis_aux_list) if loss_dis_aux_list else 0
        results['dis_aux_acc_s'] = np.mean(d_aux_acc_s) if d_aux_acc_s else 0
        results['dis_aux_acc_t'] = np.mean(d_aux_acc_t) if d_aux_acc_t else 0

        return results


    def train(self):
        """
        Main training loop
        """
        self.prepare_losses() # Prepare MPCL loss functions

        # --- Load initial source centroids ---
        try:
            source_modality = "bssfp" if "mscmrseg" in self.args.data_dir else "ct"
            centroid_filename = f'class_center_{source_modality}_f{self.args.fold}.npy'

            # Assume the centroid file is in the current working directory OR
            # the project root if running from outside
            init_centroid_path = Path(centroid_filename) # Try CWD first
            if not init_centroid_path.is_file():
                 # Try path relative to potential project root in /kaggle/working
                 # Ensure this path is correct for your Kaggle setup
                 project_dir_in_working = Path("/kaggle/working/Soft-Labeled-Contrastive-Learning/")
                 init_centroid_path = project_dir_in_working / centroid_filename

            print(f"Attempting to load initial centroids from: {init_centroid_path.resolve()}")
            if not init_centroid_path.is_file():
                raise FileNotFoundError(f"Centroid file not found at expected locations: {Path(centroid_filename).resolve()} or {init_centroid_path.resolve()}")

            self.centroid_s = np.load(init_centroid_path)
            self.centroid_s = torch.from_numpy(self.centroid_s).float().to(self.device)
            print(f"Initial centroids loaded successfully, shape: {self.centroid_s.shape}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("Please ensure the centroid file exists and the script is run from the correct directory.")
            sys.exit(1) # Now sys should be defined
        except Exception as e:
            print(f"Error loading initial centroids: {e}")
            sys.exit(1) # Now sys should be defined
        # --- End Centroid Loading ---


        for epoch in tqdm(range(self.start_epoch, self.args.epochs), desc="Training Epochs"):
            epoch_start = datetime.now()
            self.adjust_lr(epoch=epoch) # Adjust LR for segmentor and discriminators

            train_results = self.train_epoch(epoch)

            # --- Evaluation ---
            results_valid = self.eval(modality='target', phase='valid', toprint=False)
            valid_dice_scores = [results_valid['dc'][k] for k in range(1, self.args.num_classes)]
            lge_dice = np.nanmean(valid_dice_scores) if valid_dice_scores else 0.0
            lge_dice = np.round(lge_dice, 4)

            test_dice_str = ""
            if self.args.evalT:
                 results_test = self.eval(modality='target', phase='test', toprint=False)
                 test_dice_scores = [results_test['dc'][k] for k in range(1, self.args.num_classes)]
                 lge_dice_test = np.nanmean(test_dice_scores) if test_dice_scores else 0.0
                 lge_dice_test = np.round(lge_dice_test, 4)
                 test_dice_str = f", Test Dice: {lge_dice_test:.4f}"


            # --- Logging (TensorBoard and Print) ---
            print("\nWriting summary...")
            if self.args.evalT:
                self.writer.add_scalars('Dice/LGE', {'Valid': lge_dice, 'Test': lge_dice_test}, epoch + 1)
            else:
                self.writer.add_scalar('Dice/LGE_valid', lge_dice, epoch + 1)
            self.writer.add_scalar('Loss/Seg_Source', train_results['seg_s'], epoch + 1)
            self.writer.add_scalar('Loss/Seg_Source_Aux', train_results['seg_s_aux'], epoch + 1)
            self.writer.add_scalars('Loss/MPSCL', {'Source': train_results['loss_mpscl_tr'],
                                                   'Target': train_results['loss_mpscl_tg']}, epoch + 1)
            self.writer.add_scalar('Loss/CNR', train_results['loss_cnr'], epoch + 1)
            self.writer.add_scalars('Loss/Adv', {'main': train_results['loss_adv'],
                                                 'aux': train_results['loss_adv_aux']}, epoch + 1)
            self.writer.add_scalars('Loss/Dis', {'main': train_results['loss_dis'],
                                                 'aux': train_results['loss_dis_aux']}, epoch + 1)
            self.writer.add_scalars('Acc/Dis_main', {'source': train_results['dis_acc_s'],
                                                  'target': train_results['dis_acc_t']}, epoch + 1)
            self.writer.add_scalars('Acc/Dis_aux', {'source': train_results['dis_aux_acc_s'],
                                                 'target': train_results['dis_aux_acc_t']}, epoch + 1)
            self.writer.add_scalars('LR', {'Segmentor': self.opt.param_groups[0]['lr'],
                                          'Discriminator': self.opt_d.param_groups[0]['lr']}, epoch + 1)

            # Print epoch summary
            message = (f'\nEpoch = {epoch + 1:4d}/{self.args.epochs:4d} | '
                       f'LR={self.opt.param_groups[0]["lr"]:.2e} | '
                       f'Seg S={train_results["seg_s"]:.4f} | '
                       f'MPCL S={train_results["loss_mpscl_tr"]:.4f} | '
                       f'MPCL T={train_results["loss_mpscl_tg"]:.4f} | '
                       f'CNR={train_results["loss_cnr"]:.4f} | ' # Added CNR
                       f'Adv={train_results["loss_adv"]:.4f} | '
                       f'Dis={train_results["loss_dis"]:.4f} | '
                       f'Val Dice={lge_dice:.4f}{test_dice_str}')
            print(message)


            # --- Checkpointing and Early Stopping ---
            tobreak = self.stop_training(epoch, epoch_start, lge_dice)

            self.mcp_segmentor.step(monitor=lge_dice, model=self.segmentor, epoch=epoch + 1,
                                    optimizer=self.opt, tobreak=tobreak)
            self.modelcheckpoint_d.step(monitor=lge_dice, model=self.d_main, epoch=epoch + 1,
                                        optimizer=self.opt_d, tobreak=tobreak)
            if self.args.multilvl:
                self.modelcheckpoint_d_aux.step(monitor=lge_dice, model=self.d_aux, epoch=epoch + 1,
                                                optimizer=self.opt_d_aux, tobreak=tobreak)
            if tobreak:
                print(f"Stopping training at epoch {epoch+1} due to early stopping or time limit.")
                break

        self.writer.close()
        best_epoch = self.mcp_segmentor.epoch
        best_score = self.mcp_segmentor.best_result

        # --- MODIFIED LOG RENAMING ---
        # Ensure log_dir attribute exists and convert string path to Path object
        if hasattr(self, 'log_dir') and isinstance(self.log_dir, str): # Check if it's a string
            current_log_path = Path(self.log_dir) # Convert string to Path object
            if current_log_path.exists(): # Now check if the Path exists
                try:
                    # Construct new name using the Path object's parent
                    # Use 4 decimal places for score in filename
                    log_dir_new_name = '{}.e{}.Scr{:.4f}'.format(self.apdx, best_epoch, best_score)
                    log_dir_new = current_log_path.parent / log_dir_new_name
                    os.rename(current_log_path, log_dir_new) # Use os.rename with paths/strings
                    print(f"Renamed log directory to: {log_dir_new}")
                except OSError as e:
                    print(f"Error renaming log directory from {current_log_path} to {log_dir_new_name}: {e}")
                except Exception as e: # Catch other potential errors during formatting/renaming
                     print(f"An unexpected error occurred during log renaming: {e}")
            else:
                print(f"Log directory path does not exist: {current_log_path}")
        elif hasattr(self, 'log_dir'):
             print(f"Log directory ('self.log_dir') is not a string. Type: {type(self.log_dir)}. Skipping rename.")
        else:
             print("Log directory ('self.log_dir') attribute not found, skipping rename.")
        # --- END MODIFIED LOG RENAMING ---


        # --- Final Evaluation ---
        model_name = self.mcp_segmentor.best_model_save_dir
        print(f"\nLoading best model from: {model_name}")
        if model_name and os.path.exists(model_name):
             try:
                 checkpoint = torch.load(model_name)
                 if 'model_state_dict' in checkpoint:
                      self.segmentor.load_state_dict(checkpoint['model_state_dict'])
                 else:
                      self.segmentor.load_state_dict(checkpoint)
                 print("Best model loaded successfully for final evaluation.")
                 print("\n--- Final Test Evaluation (Target Domain) ---")
                 self.eval(modality='target', phase='test', toprint=True) # Print full test details
                 print("\n--- Final Test Evaluation (Source Domain) ---")
                 self.eval(modality='source', phase='test', toprint=True) # Evaluate on source test set as well
             except Exception as e:
                 print(f"Error loading best model weights for final evaluation: {e}")
        else:
             print("Best model checkpoint not found or not saved. Skipping final evaluation.")

        return