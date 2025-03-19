import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import logging

'''
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Main Idea of Code: 
- Defines the data loading pipeline for the MIMIC Chest X-ray Dataset using PyTorch and TensorFlow. 
- Loads & procesesses the image dataset embeddings (stored in TFRecod format) & prepares them for deep learning models by creating PyTorch dataset objects for model training.
- Manages GPU settings. 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

# ---- GPU Configuration - Make both GPUs visible to TensorFlow and PyTorch ----
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Import TensorFlow after setting CUDA_VISIBLE_DEVICES
import tensorflow as tf

# Print GPU information to confirm visibility
physical_devices = tf.config.list_physical_devices('GPU')
print(f"TensorFlow sees {len(physical_devices)} GPUs: {physical_devices}")
print(f"PyTorch sees {torch.cuda.device_count()} GPUs")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Configure TensorFlow to grow memory as needed
for device in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(device, True)
        print(f"Set memory growth for {device}")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")


# ---- Configure logging ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---- Image Embedding Loading ----
def load_tf_record(file_path, base_path, embedding_size=1376):
    """
    Loads image embeddings from the TFRecord files.
    Loads a TensorFlow record file with explicit GPU control.
    
    Args:
        file_path: Path to the TFRecord file
        base_path: Base path to prepend (empty string if path is already complete)
        embedding_size: Size of the embedding vector to expect
    
    Returns:
        numpy.ndarray: Embedding values or zeros if loading fails
    """
    
    full_path = file_path if not base_path else f"{base_path}/{file_path}"
    if not os.path.exists(full_path):
        logger.warning(f"File not found: {full_path}")
        return np.zeros(embedding_size, dtype=np.float32)
        
    try:
        # Configure dataset options with only the safe options
        options = tf.data.Options()
        options.experimental_optimization.apply_default_optimizations = False
        # Remove the problematic autotune setting
        
        # Read the file with the specified options
        raw_dataset = tf.data.TFRecordDataset([full_path], num_parallel_reads=1)
        raw_dataset = raw_dataset.with_options(options)
        
        embedding_values = None
        for raw_record in raw_dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            embedding_values = np.array(example.features.feature['embedding'].float_list.value, dtype=np.float32)
        
        # Explicit cleanup
        del raw_dataset
        
        # Check if the embedding is empty
        if embedding_values is None or len(embedding_values) == 0:
            return np.zeros(embedding_size, dtype=np.float32)
        
        return embedding_values
    except Exception as e:
        logger.warning(f"Error loading embedding for {file_path}: {e}")
        return np.zeros(embedding_size, dtype=np.float32)


# ---- Handles Dataset Samples - Loads, Reads, Retrieves, Support Transformations ----
class MIMICDataset(Dataset):
    def __init__(self, data_df, base_path="home/ahmedyra/scratch/Dataset/", transform=None):
        """
        Initialize the MIMIC dataset
        
        Args:
            data_df: DataFrame containing the data (either path to pickle or DataFrame)
            base_path: Base path to the dataset files
            transform: Optional transforms to apply
        """
        # Allow for both DataFrame and path to pickle
        if isinstance(data_df, str):
            self.data_df = pd.read_pickle(data_df)
        else:
            self.data_df = data_df
            
        self.transform = transform
        # Ensure base_path ends with a slash for consistent path joining
        self.base_path = base_path if base_path.endswith('/') else f"{base_path}/"
        
        self.labels = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                       'Support Devices', 'No Finding']
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get the row from the dataframe
        row = self.data_df.iloc[idx]
        
        # Get the embedding
        try:
            # Determine if this is a test sample (it will have age_decile instead of anchor_age)
            is_test = 'age_decile' in row and 'anchor_age' not in row
            
            # For test data, include the special directory
            if is_test:
                file_path = f"{self.base_path}generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0/{row['path']}"
                # Don't pass base_path again to load_tf_record
                embedding_values = load_tf_record(file_path, "")
            # Avoid duplicating the base path if it's already in the path
            elif self.base_path in row['path']:
                file_path = row['path']
                embedding_values = load_tf_record(file_path, "")
            else:
                file_path = f"{self.base_path}{row['path']}"
                embedding_values = load_tf_record(file_path, "")
        except Exception as e:
            logger.warning(f"Error loading embedding for {row['path']}: {e}")
            # Fallback to zeros if there's an error
            embedding_values = np.zeros(1376, dtype=np.float32)
        
        # Get the labels
        labels = row[self.labels].values.astype(np.float32)
        
        # Create sample dictionary
        sample = {
            'embedding': torch.tensor(embedding_values, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.float32),
            'gender': row['gender'],
            'anchor_age': row.get('anchor_age', row.get('age_decile')),  # Use anchor_age if available, otherwise use age_decile
            'insurance': row['insurance'],
            'race': row['race'],
            'patient_id': row['subject_id'],
            'study_id': row['study_id'],
            'dicom_id': row['dicom_id'],
            'path': row['path']
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def __len__(self):
        return len(self.data_df)


# --- Loads, Cleans Splits dataset for Training & Validation ---
def load_and_prepare_data(train_pickle_path, test_csv_path, val_ratio=0.1, random_state=42):
    """
    Load training data from pickle and test data from CSV, and create train/val splits
    while ensuring no test subjects appear in train/val.
    
    Note: The function handles the different column names for age between train and test datasets.
    Train data uses 'anchor_age' while test data uses 'age_decile'.
    
    Args:
        train_pickle_path: Path to the training data pickle file
        test_csv_path: Path to the test data CSV file
        val_ratio: Ratio of validation set size
        random_state: Random seed for reproducibility
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Load training data
    train_df = pd.read_pickle(train_pickle_path)
    
    # Load test data
    test_df = pd.read_csv(test_csv_path)
    
    # Get unique subject IDs from test set
    test_subject_ids = set(test_df['subject_id'].unique())
    
    # Remove any subjects from train_df that are in the test set
    train_df = train_df[~train_df['subject_id'].isin(test_subject_ids)]
    
    # Now split remaining data into train and validation
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Get unique subjects for train data
    train_subjects = train_df['subject_id'].unique()
    np.random.shuffle(train_subjects)
    
    # Calculate split point
    val_size = int(len(train_subjects) * val_ratio)
    val_subjects = train_subjects[:val_size]
    train_subjects = train_subjects[val_size:]
    
    # Split dataframes by subject
    val_df = train_df[train_df['subject_id'].isin(val_subjects)]
    train_df = train_df[train_df['subject_id'].isin(train_subjects)]
    
    logger.info(f"Train set: {len(train_df)} samples, {len(train_subjects)} subjects")
    logger.info(f"Validation set: {len(val_df)} samples, {len(val_subjects)} subjects")
    logger.info(f"Test set: {len(test_df)} samples, {len(test_subject_ids)} subjects")
    
    return train_df, val_df, test_df


# ---- Creates Dataset Objects ----
def create_datasets(train_df, val_df, test_df, base_path="home/ahmedyra/scratch/Dataset/", transform=None):
    """
    Create PyTorch Dataset objects from dataframes
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        base_path: Base path to the dataset files
        transform: Optional transforms to apply
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = MIMICDataset(train_df, base_path, transform)
    val_dataset = MIMICDataset(val_df, base_path, transform)
    test_dataset = MIMICDataset(test_df, base_path, transform)
    
    return train_dataset, val_dataset, test_dataset


# ---- Label Column Values ----
def get_label_columns():
    """
    Return the list of label column names for the MIMIC chest X-ray dataset
    
    Returns:
        list: List of label column names
    """
    return [
        'Enlarged Cardiomediastinum', 
        'Cardiomegaly', 
        'Lung Opacity',
        'Lung Lesion', 
        'Edema', 
        'Consolidation', 
        'Pneumonia', 
        'Atelectasis',
        'Pneumothorax', 
        'Pleural Effusion', 
        'Pleural Other', 
        'Fracture',
        'Support Devices', 
        'No Finding'
    ]