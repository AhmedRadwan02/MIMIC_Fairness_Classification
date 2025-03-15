# MIMIC Chest X-ray Classification

This project processes MIMIC Chest X-ray embeddings and trains a multi-label classifier to detect various chest conditions from X-ray image embeddings.

## Project Structure
- `data_loader.py`: Dataset loading and preprocessing
- `model.py`: Neural network model definition
- `train.py`: Training loop and utilities
- `evaluate.py`: Evaluation and metrics calculation
- `main.py`: Main script that ties everything together

## Requirements
- Python 3.7+
- PyTorch
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- tqdm

## Dataset Structure
This code expects:
1. Directory with MIMIC image embeddings
2. Preprocessed PKL data file containing dataset information

## Usage

### Training a new model
```bash
python main.py \
  --embedding-path /path/to/embedding/directory/ \
  --data-path-pkl /path/to/preprocessed_data.pkl \
  --output-dir results \
  --batch-size 256 \
  --epochs 10 \
  --learning-rate 0.001
```

### Using existing split information
```bash
python main.py \
  --embedding-path /path/to/embedding/directory/ \
  --data-path-pkl /path/to/preprocessed_data.pkl \
  --processed-data-path /path/to/data_with_splits.csv \
  --output-dir results \
  --batch-size 256 \
  --epochs 10
```

### Evaluating a pre-trained model
```bash
python main.py \
  --embedding-path /path/to/embedding/directory/ \
  --data-path-pkl /path/to/preprocessed_data.pkl \
  --load-model path/to/chest_xray_model.pth \
  --eval-only \
  --output-dir evaluation_results
```

### Load a pre-trained model and continue training
```bash
python main.py \
  --embedding-path /path/to/embedding/directory/ \
  --data-path-pkl /path/to/preprocessed_data.pkl \
  --load-model path/to/chest_xray_model.pth \
  --output-dir continued_training \
  --batch-size 256 \
  --epochs 10 \
  --learning-rate 0.001
```

## Example Command

```bash
python main.py --embedding-path /home/ahmedyra/scratch/Dataset/ --data-path-pkl /home/ahmedyra/projects/def-hinat/ahmedyra/EECS_Fairness_Project/preprocessed_data.pkl --output-dir results
```

## Model Architecture
The chest X-ray classifier is implemented in the `ChestXrayClassifier` class, which processes MIMIC chest X-ray image embeddings with a dimension of 1376.

## Output
The model outputs:
- Training loss and accuracy plots
- Prediction CSV files with binary prediction for each condition
- Metrics summary with performance statistics for each condition
- Saved model weights

## Command Line Arguments

### Data Paths
- `--embedding-path`: Base path to MIMIC embedding files (required)
- `--data-path-pkl`: Path to the preprocessed PKL data file (required)
- `--processed-data-path`: Path to preprocessed data CSV with split information (optional)

### Output Configuration
- `--output-dir`: Directory to save outputs (default: "output")

### Training Parameters
- `--batch-size`: Batch size for training and evaluation (default: 256)
- `--num-workers`: Number of worker processes for data loading (default: 4)
- `--epochs`: Number of epochs to train (default: 10)
- `--learning-rate`: Learning rate for optimizer (default: 0.001)
- `--val-size`: Fraction of data to use for validation (default: 0.1)
- `--test-size`: Fraction of data to use for testing (default: 0.2)
- `--seed`: Random seed for reproducibility (default: 42)
- `--checkpoint-interval`: Save predictions every N epochs (default: 5)

### Flags
- `--eval-only`: Only run evaluation (no training)
- `--no-cuda`: Disable CUDA even if available
- `--load-model`: Path to pre-trained model to load