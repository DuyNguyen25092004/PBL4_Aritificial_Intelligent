"""Script to automatically save weights and logs to Google Drive after training
"""

import os
import shutil
from datetime import datetime
from pathlib import Path


def save_to_drive(
    model_name: str = "imle_net",
    drive_path: str = "/content/drive/MyDrive/IMLE-Net-Project/results",
    include_logs: bool = True,
    include_checkpoints: bool = True,
    create_timestamp_folder: bool = True,
) -> None:
    """
    Save training results (weights and logs) to Google Drive.
    
    Parameters
    ----------
    model_name : str
        Name of the model (e.g., 'imle_net', 'mousavi', 'rajpurkar')
    drive_path : str
        Path to Google Drive folder where results will be saved
    include_logs : bool
        Whether to save training logs
    include_checkpoints : bool
        Whether to save model checkpoints
    create_timestamp_folder : bool
        Whether to create a timestamped subfolder
    
    Returns
    -------
    None
    """
    
    # Get current directory
    current_dir = os.getcwd()
    
    # Create timestamp folder if requested
    if create_timestamp_folder:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(drive_path, f"{model_name}_{timestamp}")
    else:
        save_path = drive_path
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    print("="*70)
    print("SAVING RESULTS TO GOOGLE DRIVE")
    print("="*70)
    print(f"Destination: {save_path}")
    print()
    
    saved_files = []
    
    # Save checkpoints
    if include_checkpoints:
        checkpoint_dir = os.path.join(current_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            print("📦 Saving checkpoints...")
            
            # Create checkpoints folder in Drive
            drive_checkpoint_dir = os.path.join(save_path, "checkpoints")
            os.makedirs(drive_checkpoint_dir, exist_ok=True)
            
            # Copy all checkpoint files
            for file in os.listdir(checkpoint_dir):
                if model_name in file:
                    src = os.path.join(checkpoint_dir, file)
                    dst = os.path.join(drive_checkpoint_dir, file)
                    
                    shutil.copy2(src, dst)
                    file_size = os.path.getsize(src) / (1024 * 1024)  # MB
                    print(f"  ✓ {file} ({file_size:.2f} MB)")
                    saved_files.append(file)
        else:
            print("⚠️  No checkpoints found to save")
    
    # Save logs
    if include_logs:
        logs_dir = os.path.join(current_dir, "logs")
        if os.path.exists(logs_dir):
            print("\n📊 Saving logs...")
            
            # Create logs folder in Drive
            drive_logs_dir = os.path.join(save_path, "logs")
            os.makedirs(drive_logs_dir, exist_ok=True)
            
            # Copy all log files
            for file in os.listdir(logs_dir):
                if model_name in file:
                    src = os.path.join(logs_dir, file)
                    dst = os.path.join(drive_logs_dir, file)
                    
                    shutil.copy2(src, dst)
                    file_size = os.path.getsize(src) / 1024  # KB
                    print(f"  ✓ {file} ({file_size:.2f} KB)")
                    saved_files.append(file)
        else:
            print("⚠️  No logs found to save")
    
    # Create a summary file
    summary_path = os.path.join(save_path, "save_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Training Results Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Location: {save_path}\n\n")
        f.write(f"Files saved:\n")
        for file in saved_files:
            f.write(f"  - {file}\n")
    
    print(f"\n📝 Summary saved to: save_summary.txt")
    print("="*70)
    print("✅ ALL RESULTS SAVED SUCCESSFULLY TO GOOGLE DRIVE!")
    print("="*70)
    print(f"\nYou can find your results at:\n{save_path}")
    print()


def save_weights_only(
    model_name: str = "imle_net",
    drive_path: str = "/content/drive/MyDrive/IMLE-Net-Project/results",
) -> None:
    """
    Quick function to save only weights to Google Drive.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    drive_path : str
        Path to Google Drive folder
    """
    save_to_drive(
        model_name=model_name,
        drive_path=drive_path,
        include_logs=False,
        include_checkpoints=True,
        create_timestamp_folder=False,
    )


def save_all_models(
    drive_path: str = "/content/drive/MyDrive/IMLE-Net-Project/results",
) -> None:
    """
    Save all trained models to Google Drive.
    
    Parameters
    ----------
    drive_path : str
        Path to Google Drive folder
    """
    models = ["imle_net", "mousavi", "rajpurkar", "ecgnet", "resnet101"]
    
    print("="*70)
    print("SAVING ALL MODELS TO GOOGLE DRIVE")
    print("="*70)
    
    for model in models:
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
        
        # Check if model weights exist
        weight_files = [
            f"{model}_weights.weights.h5",
            f"{model}_weights.pt",
            f"{model}_sub_diagnostic_weights.weights.h5",
        ]
        
        model_exists = False
        if os.path.exists(checkpoint_dir):
            for weight_file in weight_files:
                if os.path.exists(os.path.join(checkpoint_dir, weight_file)):
                    model_exists = True
                    break
        
        if model_exists:
            print(f"\n📦 Saving {model}...")
            save_to_drive(
                model_name=model,
                drive_path=drive_path,
                include_logs=True,
                include_checkpoints=True,
                create_timestamp_folder=False,
            )
        else:
            print(f"⏭️  Skipping {model} (not trained)")
    
    print("\n" + "="*70)
    print("✅ ALL AVAILABLE MODELS SAVED!")
    print("="*70)


if __name__ == "__main__":
    """
    Example usage:
    
    # After training, run this script:
    python save_to_drive.py
    
    # Or import and use in your training script:
    from save_to_drive import save_to_drive
    
    # Train your model
    train(model, ...)
    
    # Save to Drive
    save_to_drive(model_name="imle_net")
    """
    
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Save training results to Google Drive")
    parser.add_argument(
        "--model",
        type=str,
        default="imle_net",
        help="Model name (imle_net, mousavi, rajpurkar, ecgnet, resnet101)",
    )
    parser.add_argument(
        "--drive_path",
        type=str,
        default="/content/drive/MyDrive/IMLE-Net-Project/results",
        help="Google Drive path to save results",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Save all trained models",
    )
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Don't save logs",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Don't create timestamp folder",
    )
    # Add -f argument for Jupyter/Colab compatibility
    parser.add_argument(
        "-f",
        type=str,
        default="",
        help=argparse.SUPPRESS,  # Hide this argument from help
    )
    
    args = parser.parse_args()
    
    if args.all:
        save_all_models(drive_path=args.drive_path)
    else:
        save_to_drive(
            model_name=args.model,
            drive_path=args.drive_path,
            include_logs=not args.no_logs,
            include_checkpoints=True,
            create_timestamp_folder=not args.no_timestamp,
        )