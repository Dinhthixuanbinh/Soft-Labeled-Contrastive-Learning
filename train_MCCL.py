# %%writefile /kaggle/working/Soft-Labeled-Contrastive-Learning/train_MCCL.py
from datetime import datetime
import argparse # Ensure argparse is imported if not already
from trainer.Trainer_MCCL import Trainer_MCCL # Make sure this is the correct trainer
import config # To access any base defaults if needed before override

def main():
    trainer_mccl = Trainer_MCCL()

    # --- Hardcode Parameters for SLCL (MCCL) CT->MR Training (Fold 0) ---
    # These values will override command-line arguments and internal defaults
    print("--- WARNING: Hardcoding arguments for SLCL (MCCL) CT->MR training ---")

    # 1. Dataset and Run Configuration
    trainer_mccl.args.fold = 0
    trainer_mccl.args.split = 0 # Assuming split 0 for CT-MR
    trainer_mccl.args.dataset = 'mmwhs' # Explicitly setting for clarity
    trainer_mccl.args.data_dir = "/kaggle/input/ct-mr-2d-dataset-da/CT_MR_2D_Dataset_mmwhs" # Or from config.DATA_DIRECTORY
    trainer_mccl.args.raw_data_dir = "/kaggle/input/ct-mr-2d-dataset-da/CT_MR_2D_Dataset_mmwhs" # Or from config.RAW_DATA_DIRECTORY
    trainer_mccl.args.raw = True

    # 2. Model Configuration
    trainer_mccl.args.backbone = 'resnet50'
    trainer_mccl.args.multilvl = True # Usually beneficial
    trainer_mccl.args.phead = True    # Enable projection head for contrastive features

    # 3. Optimizer and Learning Rate
    trainer_mccl.args.optim = 'sgd'
    trainer_mccl.args.lr = 0.0008 # For CT-MR as per paper Table 3 for SLCL

    # 4. Epochs and Batch Size
    trainer_mccl.args.epochs = 200 # Example, adjust as needed
    # trainer_mccl.args.bs = 16 # You mentioned to ignore BS, but this is where you'd set it

    # 5. SLCL (MCCL) Specific Parameters (aligning with paper for CT-MR)
    trainer_mccl.args.clda = True         # Enable contrastive learning
    trainer_mccl.args.wtd_ave = True      # Enable soft-labeling (weighted average for centroids)
    trainer_mccl.args.part = 2            # Partition number for rMC
    trainer_mccl.args.inter_w = 1.0       # Lambda_CL
    trainer_mccl.args.CNR = True          # Enable Centroid Norm Regularizer
    trainer_mccl.args.CNR_w = 4e-5        # Lambda_CNR for CT-MR
    trainer_mccl.args.tau = 0.1           # Temperature for contrastive loss
    trainer_mccl.args.ctd_mmt = 0.9       # Centroid momentum
    # trainer_mccl.args.intra = True # Enable if you want intra-domain contrastive loss
    # trainer_mccl.args.intra_w = config.WEIGHT_INTRA_LOSS # If intra is enabled

    # 6. RAIN features (typically disabled for pure SLCL comparison)
    trainer_mccl.args.rain = False
    # trainer_mccl.args.update_eps = False # if rain is False, this doesn't matter much
    # trainer_mccl.args.consist_w = 0.0 # if rain is False

    # Print the final configuration being used for verification
    print(f"--- Running SLCL (MCCL) with the following effective args: ---")
    for arg_name, arg_val in sorted(vars(trainer_mccl.args).items()):
        print(f"{arg_name}: {arg_val}")
    print("----------------------------------------------------")
    # --- End Hardcoding ---

    trainer_mccl.train()

if __name__ == '__main__':
    start_time = datetime.now()
    main()
    print('Time elapsed: {}'.format(datetime.now() - start_time))
    print('program finish')