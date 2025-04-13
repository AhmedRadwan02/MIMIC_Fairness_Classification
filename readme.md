# MIMIC_Fairness_Classification

## Project Overview
This repository contains code and analysis for evaluating and improving fairness in classification models applied to the MIMIC dataset. The project focuses on debiasing various machine learning models to ensure fair predictions across different demographic groups.

## Repository Structure
- **KaggleNotebook/**: Main directory containing all notebooks and model implementations
  - **Adversarial_Debiased/**: Implementation of adversarial debiasing techniques
  - **BaseLine/**: Baseline model implementations and evaluations
  - **Ensemble_Debiased/**: Ensemble methods for model debiasing
  - **FineTuned_Baseline_Debiased/**: Fine-tuned baseline models with debiasing
  - **LWBC_Debiased/**: Loss-weighted bias correction implementations
  - **Training_Debiased_Models/**: Training notebooks and implementations
    - **Debiasing_MIMIC_Different_Models.ipynb**: Main notebook for training debiased models on MIMIC data
  - **analysis_comparison.ipynb**: Comparative analysis of different models
  - **AUC_analysis.ipynb**: Area Under Curve analysis for model performance
  - **FPR_analysis.ipynb**: False Positive Rate analysis with focus on insurance FPR
  - **Playground_analysis.ipynb**: Experimental analysis and testing
  - **TPR_analysis.ipynb**: True Positive Rate analysis and final version

## Main Training Notebook
The core of this project is found in `/KaggleNotebook/Training_Debiased_Models/Debiasing_MIMIC_Different_Models.ipynb`, which implements various debiasing techniques across different model architectures. This notebook provides a comprehensive approach to training fair models on the MIMIC dataset.

## Analysis Notebooks
Several analysis notebooks are provided to evaluate model performance:
- **TPR_analysis.ipynb**: Analyzes True Positive Rate across different demographic groups
- **FPR_analysis.ipynb**: Focuses on False Positive Rate analysis, particularly with respect to insurance coverage
- **AUC_analysis.ipynb**: Evaluates model performance using Area Under the ROC Curve
- **analysis_comparison.ipynb**: Provides comparative analysis between different model approaches

## Debiasing Approaches
This project explores multiple debiasing techniques:
1. **Adversarial debiasing**: Using adversarial networks to reduce bias
2. **Ensemble debiasing**: Combining multiple models to reduce overall bias
3. **Fine-tuned baseline debiasing**: Adjusting baseline models with debiasing objectives
4. **LWBC (Loss-Weighted Bias Correction)**: Applying weighted loss functions to correct for bias

## Getting Started
To use this repository:
1. Clone the repository
2. Navigate to the main training notebook in `/KaggleNotebook/Training_Debiased_Models/`
3. Run the notebooks in sequence, starting with baseline models and then exploring the different debiasing techniques

## Requirements
The code is designed to run in a Kaggle notebook environment, which provides all necessary dependencies including:
- Python 3.x
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Jupyter

## Results
Results from different models are compared in the analysis notebooks, with a focus on fairness metrics such as:
- Demographic parity
- Equal opportunity
- Equalized odds
- Group fairness measures

The final TPR analysis version represents the culmination of this work, showing improved fairness across demographic groups while maintaining high predictive performance.

## Contributors
- Ahmed Y. Radwan - Lassonde School of Engineering, York University, Toronto, ON, Canada (ahmedyra@yorku.ca)
- Noor Abbas - Lassonde School of Engineering, York University, Toronto, ON, Canada (nabbas24@my.yorku.ca)
- Claudia Farkas - Lassonde School of Engineering, York University, Toronto, ON, Canada (cfarkas8@my.yorku.ca)
- Amir Haeri - Lassonde School of Engineering, York University, Toronto, ON, Canada (ahaeri@yorku.ca)

## License
This project is maintained in a standard git repository with common configuration files (.gitattributes, .gitignore).
