# %%writefile /kaggle/working/Soft-Labeled-Contrastive-Learning/train_baseline.py
import sys
from pathlib import Path
from datetime import datetime
import argparse

# --- Add project root to Python path (CRITICAL: Should be at the very top) ---
project_root_str = "/kaggle/working/Soft-Labeled-Contrastive-Learning"
project_root = Path(project_root_str)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root)) # Insert at the beginning of the path
    print(f"Added {project_root} to sys.path from train_baseline.py")
else:
    print(f"{project_root} already in sys.path (train_baseline.py)")
# --- End Add project root ---

# Now import your project modules AFTER sys.path is set
from trainer.Trainer_baseline import Trainer_baseline
# config is imported within Trainer and DataGenerator, no need for explicit import here unless used directly

def main():
    trainer_base = Trainer_baseline()

    # --- Hardcode Parameters for Source-Only CT Training ---
    print("--- WARNING: Hardcoding arguments for baseline source-only CT training ---")

    trainer_base.args.train_with_s = True
    trainer_base.args.train_with_t = False
    trainer_base.args.fold = 0
    trainer_base.args.epochs = 200 # Example
    trainer_base.args.raw = True   # MMWHS raw data loader
    # trainer_base.args.backbone = 'resnet50' # This will be taken from Trainer.py default now
    trainer_base.args.pretrained = True    # Enable ImageNet pretraining for the encoder
    trainer_base.args.aug_s = True         # Enable source augmentation
    trainer_base.args.aug_t = False        # No target augmentation for source-only baseline
    trainer_base.args.aug_mode = 'simple'
    trainer_base.args.lr = 0.0008      # Specific LR for this baseline run, if different from config/Trainer default
    trainer_base.args.percent = 99.0

    # For MMWHS, ensure data_dir is set if not relying on config.py default in Trainer
    trainer_base.args.data_dir = "/kaggle/input/ct-mr-2d-dataset-da/CT_MR_2D_Dataset_mmwhs"
    trainer_base.args.raw_data_dir = "/kaggle/input/ct-mr-2d-dataset-da/CT_MR_2D_Dataset_mmwhs"


    # Print the final configuration that will be used AFTER Trainer init and these overrides
    # The effective args will be used for apdx generation inside _initialize_training_resources
    print(f"--- Intended args for baseline run (will be finalized in Trainer): ---")
    temp_args_view = {k: v for k, v in vars(trainer_base.args).items() if not k.startswith('_')}
    for arg_name, arg_val in sorted(temp_args_view.items()):
        print(f"{arg_name}: {arg_val}")
    print("----------------------------------------------------")
    
    trainer_base.train()

if __name__ == '__main__':
    start_time = datetime.now()
    main()
    print('Time elapsed: {}'.format(datetime.now() - start_time))
    print('program finish')