import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, label_columns, 
                num_epochs=25, learning_rate=0.001,
                save_dir='predictions', checkpoint_interval=5,
                device=None):
    """
    Train the model on the provided data loaders
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        label_columns: List of label column names
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        save_dir: Directory to save predictions and model
        checkpoint_interval: Interval (in epochs) to save predictions
        device: Device to train on (will detect automatically if None)
        zero_threshold: Threshold below which a vector is considered to be all zeros
    
    Returns:
        Trained model, training history (losses and metrics)
    """
    # Create directory for outputs if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda')
    logger.info(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Initialize loss and optimizer
    criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)  # Changed to 'none' to handle per-sample loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize tracking variables
    train_losses = []
    val_losses = []
    val_aucs = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        total_valid_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # Move data to device - ensure everything is on GPU
            inputs = batch['embedding'].to(device)
            targets = batch['labels'].to(device)
            
            # Detect all-zero vectors by checking if mean is exactly 0
            # This will identify vectors where all elements are exactly 0
            input_means = torch.mean(inputs, dim=1)
            valid_mask = input_means != 0
            
            # If entire batch is invalid, skip this batch
            if not torch.any(valid_mask):
                continue
                
            # Get valid samples
            valid_inputs = inputs[valid_mask]
            valid_targets = targets[valid_mask]
            
            # Track valid samples
            valid_count = valid_mask.sum().item()
            total_valid_samples += valid_count
            
            if valid_count < inputs.size(0):
                logger.info(f"Batch contains {inputs.size(0) - valid_count} zero vectors out of {inputs.size(0)} samples - ignoring them")
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass with valid samples only
            outputs = model(valid_inputs)
            # Calculate loss for valid samples
            losses = criterion(outputs, valid_targets)
            loss = losses.mean()
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * valid_count
            progress_bar.set_postfix({'loss': loss.item(), 'valid_samples': valid_count})
        
        # Adjust for actually processed samples
        if total_valid_samples > 0:
            train_loss /= total_valid_samples
        train_losses.append(train_loss)
        logger.info(f"Training processed {total_valid_samples} valid samples out of {len(train_loader.dataset)} total")
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_targets = []
        total_valid_val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device with non_blocking=True for potential speedup
                inputs = batch['embedding'].to(device, non_blocking=True)
                targets = batch['labels'].to(device, non_blocking=True)
                
                # Detect all-zero vectors by checking if mean is exactly 0
                input_means = torch.mean(inputs, dim=1)
                valid_mask = input_means != 0
                
                # If entire batch is invalid, skip
                if not torch.any(valid_mask):
                    logger.warning(f"Validation: Skipping batch: all {inputs.size(0)} samples are zero vectors")
                    continue
                
                # Get valid samples
                valid_inputs = inputs[valid_mask]
                valid_targets = targets[valid_mask]
                
                # Track valid samples
                valid_count = valid_mask.sum().item()
                total_valid_val_samples += valid_count
                
                outputs = model(valid_inputs)
                losses = criterion(outputs, valid_targets)
                val_loss += losses.mean().item() * valid_count
                
                # Keep CPU-side arrays for metrics calculation
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(valid_targets.cpu().numpy())
        
        if total_valid_val_samples > 0:
            val_loss /= total_valid_val_samples
        val_losses.append(val_loss)
        
        logger.info(f"Validation processed {total_valid_val_samples} valid samples out of {len(val_loader.dataset)} total")
        
        # Calculate AUC for each label
        if all_outputs:  # Check if we have any valid outputs
            all_outputs = np.vstack(all_outputs)
            all_targets = np.vstack(all_targets)
            all_probs = 1 / (1 + np.exp(-all_outputs))  # sigmoid
            
            aucs = {}
            for i, label in enumerate(label_columns):
                # Check if there are both positive and negative examples
                if np.sum(all_targets[:, i]) > 0 and np.sum(all_targets[:, i]) < len(all_targets):
                    try:
                        aucs[label] = roc_auc_score(all_targets[:, i], all_probs[:, i])
                    except Exception as e:
                        logger.warning(f"Error calculating AUC for {label}: {e}")
            
            if aucs:
                mean_auc = np.mean(list(aucs.values()))
            else:
                mean_auc = 0.0
        else:
            mean_auc = 0.0
            
        val_aucs.append(mean_auc)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Validation Loss: {val_loss:.4f}")
        logger.info(f"  Mean AUC: {mean_auc:.4f}")
        
        # Save model at specified intervals or if it's the best model so far
        if (epoch + 1) % checkpoint_interval == 0 or epoch == num_epochs - 1:
            # Save model
            model_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f"Checkpoint saved to {model_path}")
            
            # Save predictions
            try:
                from evaluate import save_predictions_to_csv
                
                train_probs, train_targets = save_predictions_to_csv(
                    train_loader, model, device, label_columns, 
                    os.path.join(save_dir, f'train_predictions_epoch_{epoch+1}.csv')
                )
                
                val_probs, val_targets = save_predictions_to_csv(
                    val_loader, model, device, label_columns, 
                    os.path.join(save_dir, f'val_predictions_epoch_{epoch+1}.csv')
                )
            except Exception as e:
                logger.warning(f"Error saving predictions: {e}")
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'chest_xray_model.pth'))
    logger.info(f"Model saved to {os.path.join(save_dir, 'chest_xray_model.pth')}")
    
    # Return model and history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_aucs': val_aucs
    }
    
    return model, history

def plot_training_history(history, save_path=None):
    """
    Plot the training history
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save plot (if None, plot will be shown)
    """
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    # Plot AUC
    plt.subplot(1, 2, 2)
    plt.plot(history['val_aucs'], label='Mean AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Validation AUC')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")
    else:
        plt.show()

def set_gpu_environment():
    """
    Setup optimal GPU environment for training
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Running on CPU.")
        return False
    
    # Log GPU information
    device_count = torch.cuda.device_count()
    logger.info(f"Found {device_count} CUDA device(s)")
    
    for i in range(device_count):
        device_properties = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {device_properties.name} - {device_properties.total_memory / 1e9:.2f} GB memory")
    
    # Set CUDA benchmark mode for optimal performance
    torch.backends.cudnn.benchmark = True
    
    # Set deterministic mode if needed for reproducibility
    # Note: This can slow down training
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = True
    
    # Set the current device
    current_device = torch.cuda.current_device()
    logger.info(f"Using GPU {current_device}: {torch.cuda.get_device_name(current_device)}")
    
    return True