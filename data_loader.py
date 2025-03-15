import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force TensorFlow to CPU
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Make GPU 0 visible again


import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

class MIMICDataset(Dataset):
    """
    A PyTorch Dataset for the MIMIC Chest X-ray data with embeddings.
    """
    def __init__(self, data_path, base_path="/home/ahmedyra/scratch/Dataset/", transform=None):
        """
        Args:
            data_path (str): Path to the pickle file containing the preprocessed data
            base_path (str): Base path to the embeddings
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_df = pd.read_pickle(data_path)
        self.transform = transform
        self.base_path = base_path
        self.labels = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                       'Support Devices', 'No Finding']
        self.demographic = ['gender', 'insurance', 'anchor_age', 'race']
        
        # Track statistics about embeddings
        self.loaded_count = 0
        self.error_count = 0
    
    def debug_tf_record(self, file_path):
        """Debug a TensorFlow record file by printing its structure"""
        full_path = f"{self.base_path}/{file_path}"
        if not os.path.exists(full_path):
            print(f"File does not exist: {full_path}")
            return None
            
        try:
            raw_dataset = tf.data.TFRecordDataset(full_path)
            for raw_record in raw_dataset.take(1):
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                print("Features available in TFRecord:")
                for key in example.features.feature:
                    feature = example.features.feature[key]
                    if feature.HasField('float_list'):
                        print(f"  {key}: float_list with {len(feature.float_list.value)} values")
                    elif feature.HasField('int64_list'):
                        print(f"  {key}: int64_list with {len(feature.int64_list.value)} values")
                    elif feature.HasField('bytes_list'):
                        print(f"  {key}: bytes_list with {len(feature.bytes_list.value)} values")
                return example
        except Exception as e:
            print(f"Error examining TFRecord file {full_path}: {e}")
            return None
    
    def __read_tf_record__(self, file_path):
        """Read a TensorFlow record file and parse the example"""
        full_path = f"{self.base_path}/{file_path}"
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File does not exist: {full_path}")
            
        raw_dataset = tf.data.TFRecordDataset(full_path)
        for raw_record in raw_dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            return example
            
    def __len__(self):
        """Return the total number of samples"""
        return len(self.data_df)
        
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to fetch
        Returns:
            dict: A dictionary containing the data sample
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get the row from the dataframe
        row = self.data_df.iloc[idx]
        
        # Get the embedding
        try:
            example = self.__read_tf_record__(row['path'])
            # Extract embedding values from float_list
            embedding_values = np.array(example.features.feature['embedding'].float_list.value, dtype=np.float32)
            self.loaded_count += 1
            
            # Check if the embedding is empty
            if len(embedding_values) == 0:
                print(f"Warning: Empty embedding for {row['path']}")
                embedding_values = np.zeros(1376, dtype=np.float32)
                self.error_count += 1
                
        except Exception as e:
            print(f"Error loading embedding for {row['path']}: {e}")
            # Check if the file exists
            full_path = f"{self.base_path}/{row['path']}"
            if not os.path.exists(full_path):
                print(f"File does not exist: {full_path}")
                
            # Fallback to zeros if there's an error
            embedding_values = np.zeros(1376, dtype=np.float32)
            self.error_count += 1
        
        # Get the labels
        labels = row[self.labels].values.astype(np.float32)
        
        # Get the demographic data
        demographics = row[self.demographic].values
        
        # Create sample dictionary
        sample = {
            'embedding': torch.tensor(embedding_values, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.float32),
            'demographics': demographics,
            'gender': row['gender'],
            'insurance': row['insurance'],
            'anchor_age': row['anchor_age'],
            'race': row['race'],
            'study_id': row['study_id'],
            'dicom_id': row['dicom_id'],
            'path': row['path']
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def print_stats(self):
        """Print statistics about the dataset loading"""
        total = self.loaded_count + self.error_count
        if total > 0:
            success_rate = (self.loaded_count / total) * 100
            print(f"Dataset loading stats:")
            print(f"  Successfully loaded embeddings: {self.loaded_count}")
            print(f"  Errors loading embeddings: {self.error_count}")
            print(f"  Success rate: {success_rate:.2f}%")
        else:
            print("No embeddings have been loaded yet")

def create_train_val_test_split(dataset, val_ratio=0.1, test_ratio=0.2, random_state=42):
    """
    Create indices for train/validation/test split of a dataset
    
    Args:
        dataset: PyTorch Dataset object
        val_ratio: Ratio of validation set size
        test_ratio: Ratio of test set size
        random_state: Random seed for reproducibility
    
    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Get total size
    dataset_size = len(dataset)
    
    # Calculate sizes
    test_size = int(dataset_size * test_ratio)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - test_size - val_size
    
    # Create indices
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices

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
