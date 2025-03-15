import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_predictions_to_csv(dataset_loader, model, device, label_columns, file_name):
    """
    Generate predictions and save them to a CSV file including all sample information
    
    Args:
        dataset_loader: DataLoader for the dataset
        model: Trained model
        device: Device to run inference on
        label_columns: List of label column names
        file_name: Path to save the CSV file
    
    Returns:
        Tuple of (probabilities, targets)
    """
    model.eval()
    all_outputs = []
    all_targets = []
    
    # Store all sample information
    sample_info = {
        'study_id': [],
        'dicom_id': [],
        'gender': [],
        'insurance': [],
        'anchor_age': [],
        'race': [],
        'path': []
    }
    
    with torch.no_grad():
        for batch in dataset_loader:
            inputs = batch['embedding'].to(device)
            targets = batch['labels'].to(device)
            
            # Get all sample information
            sample_info['study_id'].extend(batch['study_id'])
            sample_info['dicom_id'].extend(batch['dicom_id'])
            sample_info['gender'].extend(batch['gender'])
            sample_info['insurance'].extend(batch['insurance'])
            sample_info['anchor_age'].extend(batch['anchor_age'])
            sample_info['race'].extend(batch['race'])
            sample_info['path'].extend(batch['path'])
            
            outputs = model(inputs)
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)
    all_probs = 1 / (1 + np.exp(-all_outputs))  # sigmoid
    all_binary_preds = (all_probs >= 0.5).astype(int)  # Convert to binary predictions
    
    # Create DataFrame for predictions with all sample information
    pred_df = pd.DataFrame()
    
    # Add all sample information
    for key, values in sample_info.items():
        pred_df[key] = values
    
    # Add ground truth for each disease
    for i, label in enumerate(label_columns):
        pred_df[f"{label}_true"] = all_targets[:, i]
    
    # Add predictions for each disease
    for i, label in enumerate(label_columns):
        pred_df[f"{label}_prob"] = all_probs[:, i]  # Probability predictions
        pred_df[f"{label}_pred"] = all_binary_preds[:, i]  # Binary predictions
    
    # Save to CSV
    pred_df.to_csv(file_name, index=False)
    logger.info(f"Predictions saved to {file_name}")
    
    return all_probs, all_targets

def evaluate_model(model, test_loader, label_columns, device, output_dir):
    """
    Evaluate the model and generate metrics
    
    Args:
        model: Trained model
        test_loader: DataLoader for test dataset
        label_columns: List of label column names
        device: Device to run inference on
        output_dir: Directory to save results
    
    Returns:
        Dictionary of metrics
    """
    # Generate predictions
    test_probs, test_targets = save_predictions_to_csv(
        test_loader, model, device, label_columns, 
        f'{output_dir}/test_predictions_final.csv'
    )
    
    # Convert probabilities to binary predictions
    test_preds = (test_probs >= 0.5).astype(int)
    
    # Calculate metrics for each label
    metrics = {}
    metrics_df = pd.DataFrame(columns=['Label', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1'])
    
    logger.info("Final metrics by condition:")
    for i, label in enumerate(label_columns):
        # Skip if no positive examples
        if sum(test_targets[:, i]) > 0:
            auc = roc_auc_score(test_targets[:, i], test_probs[:, i])
            accuracy = accuracy_score(test_targets[:, i], test_preds[:, i])
            precision = precision_score(test_targets[:, i], test_preds[:, i], zero_division=0)
            recall = recall_score(test_targets[:, i], test_preds[:, i], zero_division=0)
            f1 = f1_score(test_targets[:, i], test_preds[:, i], zero_division=0)
            
            # Store metrics
            metrics[label] = {
                'AUC': auc,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            }
            
            # Add to DataFrame
            metrics_df = pd.concat([
                metrics_df,
                pd.DataFrame({
                    'Label': [label],
                    'AUC': [auc],
                    'Accuracy': [accuracy],
                    'Precision': [precision],
                    'Recall': [recall],
                    'F1': [f1]
                })
            ], ignore_index=True)
            
            # Log metrics
            logger.info(f"{label}:")
            logger.info(f"  AUC: {auc:.4f}")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1: {f1:.4f}")
    
    # Calculate overall metrics
    mean_auc = metrics_df['AUC'].mean()
    mean_accuracy = metrics_df['Accuracy'].mean()
    mean_precision = metrics_df['Precision'].mean()
    mean_recall = metrics_df['Recall'].mean()
    mean_f1 = metrics_df['F1'].mean()
    
    metrics['overall'] = {
        'AUC': mean_auc,
        'Accuracy': mean_accuracy,
        'Precision': mean_precision,
        'Recall': mean_recall,
        'F1': mean_f1
    }
    
    logger.info(f"\nOverall Mean AUC: {mean_auc:.4f}")
    logger.info(f"Overall Mean Accuracy: {mean_accuracy:.4f}")
    logger.info(f"Overall Mean Precision: {mean_precision:.4f}")
    logger.info(f"Overall Mean Recall: {mean_recall:.4f}")
    logger.info(f"Overall Mean F1: {mean_f1:.4f}")
    
    # Save metrics to CSV
    metrics_df.to_csv(f'{output_dir}/metrics_summary.csv', index=False)
    logger.info(f"Metrics summary saved to '{output_dir}/metrics_summary.csv'")
    
    # Add demographic analysis
    try:
        # Load the predictions with demographic information
        preds_df = pd.read_csv(f'{output_dir}/test_predictions_final.csv')
        
        # Analyze metrics by demographic groups
        demographic_analysis(preds_df, label_columns, output_dir)
    except Exception as e:
        logger.warning(f"Could not perform demographic analysis: {e}")
    
    return metrics

def demographic_analysis(preds_df, label_columns, output_dir):
    """
    Analyze model performance across demographic groups
    
    Args:
        preds_df: DataFrame with predictions and demographic information
        label_columns: List of label column names
        output_dir: Directory to save results
    """
    demographics = ['gender', 'race', 'insurance']
    
    for demographic in demographics:
        # Skip if demographic column is not in the DataFrame
        if demographic not in preds_df.columns:
            continue
            
        logger.info(f"\nAnalyzing performance by {demographic}:")
        
        # Create a DataFrame to store metrics by demographic group
        demo_metrics = []
        
        # Get unique values for this demographic
        groups = preds_df[demographic].dropna().unique()
        
        for group in groups:
            # Filter data for this group
            group_df = preds_df[preds_df[demographic] == group]
            
            if len(group_df) < 10:  # Skip groups with too few samples
                continue
                
            group_metrics = {'Demographic Group': group}
            
            # Calculate metrics for each label
            for label in label_columns:
                true_col = f"{label}_true"
                pred_col = f"{label}_pred"
                prob_col = f"{label}_prob"
                
                # Skip if columns don't exist
                if not all(col in group_df.columns for col in [true_col, pred_col, prob_col]):
                    continue
                    
                # Skip if no positive examples
                if sum(group_df[true_col]) > 0:
                    try:
                        auc = roc_auc_score(group_df[true_col], group_df[prob_col])
                        group_metrics[f"{label}_AUC"] = auc
                    except:
                        pass
                        
                    accuracy = accuracy_score(group_df[true_col], group_df[pred_col])
                    precision = precision_score(group_df[true_col], group_df[pred_col], zero_division=0)
                    recall = recall_score(group_df[true_col], group_df[pred_col], zero_division=0)
                    f1 = f1_score(group_df[true_col], group_df[pred_col], zero_division=0)
                    
                    group_metrics[f"{label}_Accuracy"] = accuracy
                    group_metrics[f"{label}_Precision"] = precision
                    group_metrics[f"{label}_Recall"] = recall
                    group_metrics[f"{label}_F1"] = f1
            
            demo_metrics.append(group_metrics)
        
        if demo_metrics:
            # Create DataFrame with metrics by demographic group
            demo_df = pd.DataFrame(demo_metrics)
            
            # Save to CSV
            demo_file = f'{output_dir}/metrics_by_{demographic}.csv'
            demo_df.to_csv(demo_file, index=False)
            logger.info(f"Metrics by {demographic} saved to '{demo_file}'")