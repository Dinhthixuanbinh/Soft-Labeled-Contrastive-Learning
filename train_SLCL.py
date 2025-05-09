# %%writefile /kaggle/working/Soft-Labeled-Contrastive-Learning/train_SLCL.py 
from datetime import datetime
import argparse # Make sure argparse is imported
from trainer.Trainer_MPSCL import Trainer_MPSCL
def main():
    trainer_mpscl = Trainer_MPSCL()

    # --- Hardcode Parameters for SLCL CT->MR Training (Fold 0) ---
    print("--- WARNING: Hardcoding arguments for SLCL CT->MR training ---")

    # 1. Set Fold
    trainer_mpscl.args.fold = 0

    # 2. Set Epochs (e.g., 1000 from AdaptSeg.yaml, adjust as needed)
    trainer_mpscl.args.epochs = 300

    # 3. Enable Raw Data Loading (Essential for CT-MR)
    trainer_mpscl.args.raw = True

    # 4. Set Backbone and its parameters (Match your baseline)
    trainer_mpscl.args.backbone = 'resnet50'
    trainer_mpscl.args.filters = 16
    trainer_mpscl.args.nb = 4
    trainer_mpscl.args.bd = 4

    # 5. Enable Multilevel (Recommended for adaptation trainers)
    trainer_mpscl.args.multilvl = True

    # 6. Set Learning Rate (Paper value for CT-MR)
    trainer_mpscl.args.lr = 0.0008

    # 7. Set Partition Number (Paper value for rMC)
    #    WARNING: Ensure -part argument definition and logic exist in Trainer_MPSCL!
    try:
        trainer_mpscl.args.part = 2
    except AttributeError:
        print("WARNING: args.part attribute does not exist. Add '-part' argument to Trainer_MPSCL.py.")
        # Attempt to add dynamically (may not work depending on when it's needed)
        # setattr(trainer_mpscl.args, 'part', 2)

    # 8. Set CNR Weight (Paper value for CT-MR)
    #    WARNING: Ensure -CNR_w argument definition and logic exist in Trainer_MPSCL!
    try:
        trainer_mpscl.args.CNR_w = 0.00004
    except AttributeError:
        print("WARNING: args.CNR_w attribute does not exist. Add '-CNR_w' argument to Trainer_MPSCL.py.")
        # Attempt to add dynamically
        # setattr(trainer_mpscl.args, 'CNR_w', 0.00004)
        # Also ensure logic to *use* args.CNR_w exists in train_epoch


    # 9. Verify Data Paths (Uses defaults from config.py - ensure they are correct)
    # trainer_mpscl.args.data_dir = "/kaggle/input/ct-mr-2d-dataset-da/CT_MR_2D_Dataset_mmwhs"
    # trainer_mpscl.args.raw_data_dir = "/kaggle/input/ct-mr-2d-dataset-da/CT_MR_2D_Dataset_mmwhs"

    # 10. Other parameters (BS, Temp, CL Weights, Momentum) use previously edited defaults
    #     No override needed here assuming previous edits to config.py/Trainer_MPSCL.py are done.

    # Print the final configuration being used
    print(f"Using Fold: {trainer_mpscl.args.fold}")
    print(f"Using Epochs: {trainer_mpscl.args.epochs}")
    print(f"Using Raw Data Mode: {trainer_mpscl.args.raw}")
    print(f"Using Backbone: {trainer_mpscl.args.backbone} (Filters: {trainer_mpscl.args.filters}, NB: {trainer_mpscl.args.nb}, BD: {trainer_mpscl.args.bd})")
    print(f"Using Multilevel: {trainer_mpscl.args.multilvl}")
    print(f"Using Learning Rate (Seg): {trainer_mpscl.args.lr}")
    print(f"Using Batch Size: {trainer_mpscl.args.bs}") # From config.py or baseline default
    print(f"Using Optimizer: {trainer_mpscl.args.optim}") # From base default
    print(f"Using Temperature: {trainer_mpscl.args.src_temp} / {trainer_mpscl.args.trg_temp}") # From MPSCL default edit
    print(f"Using MPCL Weights: S={trainer_mpscl.args.w_mpcl_s}, T={trainer_mpscl.args.w_mpcl_t}") # From MPSCL default edit
    print(f"Using Centroid Momentum: {trainer_mpscl.args.class_center_m}") # From MPSCL default edit
    print(f"Using Partition Number (P): {getattr(trainer_mpscl.args, 'part', 'Not Set/Found')}")
    print(f"Using CNR Weight (lambda_CNR): {getattr(trainer_mpscl.args, 'CNR_w', 'Not Set/Found')}")
    print(f"Using Data Directory: {trainer_mpscl.args.data_dir}")
    print("----------------------------------------------------")
    # --- End Hardcoding ---

    trainer_mpscl.train()

if __name__ == '__main__':
    start_time = datetime.now()
    main()
    print('Time elapsed: {}'.format(datetime.now() - start_time))
    print('program finish')