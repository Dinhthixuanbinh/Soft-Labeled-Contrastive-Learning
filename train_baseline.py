# %%writefile /kaggle/working/Soft-Labeled-Contrastive-Learning/train_baseline.py
from trainer.Trainer_baseline import Trainer_baseline
from datetime import datetime
import argparse # Make sure argparse is imported

def main():
    trainer_base = Trainer_baseline()

    # --- Hardcode Parameters for Source-Only CT Training ---
    # This section overrides any command-line arguments or defaults from argparse
    print("--- WARNING: Hardcoding arguments for baseline source-only CT training ---")

    # 1. Ensure Source Training is Enabled and Target is Disabled
    #    (This overrides the edit made previously in Trainer_baseline.py's add_additional_arguments)
    trainer_base.args.train_with_s = True
    trainer_base.args.train_with_t = False

    # 2. Set Fold (e.g., Fold 0 - change to 1 for the other fold)
    trainer_base.args.fold = 0

    # 3. Set Epochs (Choose a suitable number, e.g., 200)
    trainer_base.args.epochs = 200

    # 4. Enable Raw Data Loading (Crucial for CT-MR .nii files)
    trainer_base.args.raw = True

    # 5. Specify Backbone (e.g., ResNet50 as used in paper)
    trainer_base.args.backbone = 'resnet50'

    # 6. Verify Data Paths (Using defaults from config.py - ensure they are correct)
    #    You could explicitly set them here if needed:
    # trainer_base.args.data_dir = "/kaggle/input/ct-mr-2d-dataset-da/CT_MR_2D_Dataset_mmwhs"
    # trainer_base.args.raw_data_dir = "/kaggle/input/ct-mr-2d-dataset-da/CT_MR_2D_Dataset_mmwhs"

    # 7. Batch Size (Using the default 16 from config.py as requested by user)
    #    No override needed here if config.py BATCH_SIZE = 16

    # 8. Learning Rate (Using the default 0.0008 from config.py after user edits)
    #    No override needed here if config.py LEARNING_RATE = 0.0008

    # Print the hardcoded values being used for verification
    print(f"Hardcoded Fold: {trainer_base.args.fold}")
    print(f"Hardcoded Epochs: {trainer_base.args.epochs}")
    print(f"Hardcoded Raw: {trainer_base.args.raw}")
    print(f"Hardcoded train_with_s: {trainer_base.args.train_with_s}")
    print(f"Hardcoded train_with_t: {trainer_base.args.train_with_t}")
    print(f"Hardcoded Backbone: {trainer_base.args.backbone}")
    print(f"Using Data Directory: {trainer_base.args.data_dir}")
    print(f"Using Batch Size: {trainer_base.args.bs}") # Will reflect value from config.py
    print(f"Using Learning Rate: {trainer_base.args.lr}") # Will reflect value from config.py
    print("----------------------------------------------------")
    # --- End Hardcoding ---

    trainer_base.train()

if __name__ == '__main__':
    start_time = datetime.now()
    main()
    print('Time elapsed: {}'.format(datetime.now() - start_time))
    print('program finish')