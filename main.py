import torch
import argparse
import logging
import pandas as pd
from torch.utils.data import random_split, DataLoader, Subset
import matplotlib.pyplot as plt
from data_loader import MIMICDataset, create_train_val_test_split, get_label_columns
from model import ChestXrayClassifier
from train import train_model, plot_training_history
from evaluate import evaluate_model

import os
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
    ages = [item['anchor_age'] for item in batch]
    races = [item['race'] for item in batch]
    
    # For consistent numeric data, we can convert to tensors
    if all(isinstance(age, (int, float)) for age in ages):
        ages = torch.tensor(ages, dtype=torch.float32)
    
    # Create a collated batch with all fields
    collated_batch = {
        'embedding': embeddings,
        'labels': labels,
        'demographics': [item['demographics'] for item in batch],  # Keep as list of arrays
        'gender': genders,
        'insurance': insurances,
        'anchor_age': ages,
        'race': races,
        'study_id': [item['study_id'] for item in batch],
        'dicom_id': [item['dicom_id'] for item in batch],
        'path': [item['path'] for item in batch]
    }
    
    return collated_batch
def setup_gpu():
    """
    Setup and optimize GPU environment
    
    Returns:
        bool: True if GPU is available and configured
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Running on CPU.")
        return False
    
    # Log GPU information
    device_count = torch.cuda.device_count()
    logger.info(f"Found {device_count} CUDA device(s)")
    
    for i in range(device_count):
        device_properties = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {device_properties.name} - {device_properties.total_memory / 1e9:.2f} GB memory")
    
    # Enable cuDNN benchmark mode for optimized performance
    torch.backends.cudnn.benchmark = True
    logger.info("CUDA benchmark mode enabled for optimized performance")
    
    # Set current device
    current_device = torch.cuda.current_device()
    logger.info(f"Using GPU {current_device}: {torch.cuda.get_device_name(current_device)}")
    
    return True

def main(args):
    """
    Main function to run the training and evaluation pipeline
    
    Args:
        args: Command line arguments
    """
    # Setup GPU first
    if not args.no_cuda:
        gpu_available = setup_gpu()
    else:
        gpu_available = False
    
    # Check GPU availability
    device = torch.device('cuda' if gpu_available else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Pin memory if using GPU for faster data transfer
    pin_memory = (device.type == 'cuda')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.embedding_path}")
    
    # Use raw data files with MIMICDataset
    logger.info(f"Loading from PKL data file: {args.data_path_pkl}")
    dataset = MIMICDataset(
        data_path=args.data_path_pkl,
        base_path=args.embedding_path
    )

    # Print the dataset length
    print(f"Dataset length: {len(dataset)}")
    
    # Debugging - examine the first few TFRecord files
    print("\nExamining the first TFRecord file structure:")
    first_path = dataset.data_df.iloc[0]['path']
    dataset.debug_tf_record(first_path)
    
    # Get and print one sample (the first one)
    print("\nAttempting to load the first sample:")
    sample = dataset[0]
    
    print("\nSample contents:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: Tensor shape {value.shape}")
        else:
            print(f"{key}: {value}")
    
    # Print the first few values of the embedding
    print("\nFirst 10 embedding values:")
    print(sample['embedding'][:10])
    
    # Print the labels
    print("\nLabels:")
    for i, label_name in enumerate(dataset.labels):
        print(f"{label_name}: {sample['labels'][i]}")
    
    # Check if the first 10 samples load correctly
    print("\nTesting loading of the first 10 samples:")
    for i in range(10):
        try:
            _ = dataset[i]
            print(f"Sample {i}: Successfully loaded")
        except Exception as e:
            print(f"Sample {i}: Error - {e}")
    
    # Print stats after loading samples
    dataset.print_stats()
    
    if args.processed_data_path and os.path.exists(args.processed_data_path):
        # If preprocessed CSV exists with split information, use those splits
        logger.info(f"Checking split information from {args.processed_data_path}")
        data_df = pd.read_csv(args.processed_data_path)
        
        # Check if split column exists
        if 'split' in data_df.columns and not all(data_df['split'] == 'none'):
            # Use existing splits
            logger.info("Using existing splits from processed data")
            train_indices = data_df[data_df['split'] == 'train'].index.tolist()
            val_indices = data_df[data_df['split'] == 'val'].index.tolist()
            test_indices = data_df[data_df['split'] == 'test'].index.tolist()
        else:
            # Create new splits
            logger.info("No split information found, creating new train/val/test splits")
            train_indices, val_indices, test_indices = create_train_val_test_split(
                dataset,
                val_ratio=args.val_size,
                test_ratio=args.test_size,
                random_state=args.seed
            )
    else:
        # Create new splits
        logger.info("Creating new train/val/test splits")
        train_indices, val_indices, test_indices = create_train_val_test_split(
            dataset,
            val_ratio=args.val_size,
            test_ratio=args.test_size,
            random_state=args.seed
        )
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    if dataset is None or len(dataset) == 0:
        logger.error("Failed to load dataset. Exiting.")
        return
    
    logger.info(f"Dataset loaded with {len(dataset)} total samples")
    logger.info(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)}, Testing: {len(test_dataset)}")
    
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
        
        # Create data loaders with GPU optimizations
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers = 0,
            collate_fn=custom_collate
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=custom_collate,
            num_workers = 0,
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
            checkpoint_interval=args.checkpoint_interval,
            device=device
        )
        
        # Plot training history
        plot_training_history(
            history=history,
            save_path=os.path.join(args.output_dir, 'training_history.png')
        )
    else:
        logger.info("Skipping training (evaluation-only mode)")
        trained_model = model
    
    # Evaluate the model
    logger.info("Evaluating model")
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        pin_memory=pin_memory,
        collate_fn=custom_collate,
    )
    
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
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for training and evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of worker processes for data loading")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for optimizer")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Fraction of data to use for validation")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data to use for testing")
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