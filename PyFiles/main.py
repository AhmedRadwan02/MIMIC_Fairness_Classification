import torch
import argparse
import logging
import pandas as pd
from torch.utils.data import random_split, DataLoader, Subset
import matplotlib.pyplot as plt
from data_loader import MIMICDataset, get_label_columns, load_and_prepare_data, create_datasets
from model import ChestXrayClassifier
from PyFiles.train import train_model, plot_training_history
from evaluate import evaluate_model, save_predictions_to_csv
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def custom_collate(batch):
    """
    Custom collate function that handles mixed data types correctly
    """
    # Extract data for batching
    embeddings = torch.stack([item['embedding'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # For demographics and other fields, just keep them as lists
    genders = [item['gender'] for item in batch]
    insurances = [item['insurance'] for item in batch]
    
    # Handle different age column names (anchor_age for train/val, age_decile for test)
    ages = []
    for item in batch:
        if 'anchor_age' in item:
            ages.append(item['anchor_age'])
        elif 'age_decile' in item:
            ages.append(item['age_decile'])
        else:
            ages.append(None)  # Fallback
    
    races = [item['race'] for item in batch]
    
    # For consistent numeric data, we can convert to tensors
    if all(isinstance(age, (int, float)) for age in ages if age is not None):
        # Convert None values to -1 or another placeholder
        numeric_ages = [age if age is not None else -1 for age in ages]
        ages = torch.tensor(numeric_ages, dtype=torch.float32)
    
    # Create a collated batch with all fields
    collated_batch = {
        'embedding': embeddings,
        'labels': labels,
        'gender': genders,
        'insurance': insurances,
        'age': ages,  # Generic key for age regardless of source column
        'race': races,
        'patient_id': [item.get('patient_id', item.get('subject_id')) for item in batch],
        'study_id': [item['study_id'] for item in batch],
        'dicom_id': [item['dicom_id'] for item in batch],
        'path': [item['path'] for item in batch]
    }
    
    return collated_batch

def main(args):
    """
    Main function to run the training and evaluation pipeline
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device("cuda")
    logger.info(f"Using device: {device}")
    
    # Pin memory if using GPU for faster data transfer
    pin_memory = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Hard-coded path for the test CSV file
    test_csv_path = "/home/ahmedyra/projects/def-hinat/ahmedyra/EECS_Fairness_Project/mimic_test_df.csv"
    
    # Load and prepare datasets
    logger.info(f"Loading training data from {args.data_path_pkl}")
    logger.info(f"Loading test data from {test_csv_path}")
    logger.info(f"Base embedding path: {args.embedding_path}")
    
    # Load train/val/test data with subject separation
    train_df, val_df, test_df = load_and_prepare_data(
        train_pickle_path=args.data_path_pkl,
        test_csv_path=test_csv_path,
        val_ratio=args.val_size,
        random_state=args.seed
    )
    
    # Create PyTorch datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        base_path=args.embedding_path
    )
    
    if train_dataset is None or len(train_dataset) == 0:
        logger.error("Failed to load training dataset. Exiting.")
        return
    
    logger.info(f"Datasets loaded:")
    logger.info(f"  Training: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")
    logger.info(f"  Testing: {len(test_dataset)} samples")
    
    # Get label columns
    label_columns = get_label_columns()
    
    # Initialize model
    embedding_dim = 1376  # This is the size from the MIMIC embedding file
    model = ChestXrayClassifier(input_dim=embedding_dim, output_dim=len(label_columns))
    
    # Move model to GPU first, before loading state
    model = model.to(device)
    
    # Load pre-trained model if provided
    if args.load_model:
        try:
            # Use map_location to ensure model loads to the correct device
            model.load_state_dict(torch.load(args.load_model, map_location=device))
            logger.info(f"Loaded pre-trained model from {args.load_model}")
        except Exception as e:
            logger.warning(f"Failed to load model from {args.load_model}: {e}")
    
    # Train model if not in evaluation-only mode
    if not args.eval_only:
        logger.info("Starting model training")
        # Create data loaders with proper settings
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=args.num_workers,
            collate_fn=custom_collate,
            prefetch_factor=2,  # Prefetch batches
            persistent_workers=args.num_workers > 0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            num_workers=args.num_workers,
            collate_fn=custom_collate,
            persistent_workers=args.num_workers > 0
        )
        
        # Train the model
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            label_columns=label_columns,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            save_dir=args.output_dir,
            checkpoint_interval=args.checkpoint_interval
        )
        print("If you see this, execution is continuing after training")         
        # Plot training history
        plot_training_history(
            history=history,
            save_path=os.path.join(args.output_dir, 'training_history.png')
        )
    else:
        logger.info("Skipping training (evaluation-only mode)")
        trained_model = model
    # Evaluate the model on the test set
    logger.info("Evaluating model on test set")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        pin_memory=pin_memory,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        persistent_workers=args.num_workers > 0
    )
    
    # Create test predictions CSV
    test_probs, test_targets = save_predictions_to_csv(
        test_loader, 
        trained_model, 
        device, 
        label_columns,
        os.path.join(args.output_dir, 'test_predictions.csv')
    )
    
    # Evaluate the model
    metrics = evaluate_model(
        model=trained_model,
        test_loader=test_loader,
        label_columns=label_columns,
        device=device,
        output_dir=args.output_dir
    )
    
    logger.info("Evaluation complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate chest X-ray classifier")
    
    # Data paths
    parser.add_argument("--embedding-path", type=str, 
                        default="/home/ahmedyra/scratch/Dataset/",
                        help="Base path to MIMIC embedding files")
    parser.add_argument("--processed-data-path", type=str, default=None,
                        help="Path to preprocessed data CSV (optional)")
    parser.add_argument("--data-path-pkl", type=str, 
                        default="/home/ahmedyra/projects/def-hinat/ahmedyra/EECS_Fairness_Project/preprocessed_data.pkl",
                        help="Path to the preprocessed PKL data file")
    
    # Output directory
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save outputs")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training and evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of worker processes for data loading")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for optimizer")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Fraction of data to use for validation")
    parser.add_argument("--test-size", type=float, default=0.0,
                        help="Fraction of data to use for testing (ignored when using external test set)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--checkpoint-interval", type=int, default=5,
                        help="Save predictions every N epochs")
    
    # Flags
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation (no training)")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA even if available")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to pre-trained model to load")
    
    args = parser.parse_args()
    main(args)