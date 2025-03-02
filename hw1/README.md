# Advanced Machine Learning Homework 1

This repository contains Jupyter notebooks implementing and analyzing online learning algorithms and loss functions.

## Overview

The homework consists of three main components:

1. Online Learning with Expert Advice (Supervised)
   - Implementation in `q1_spambase.ipynb`
   - Uses the Spambase dataset to compare different classification algorithms
   - Implements static and fixed-share weight updates
   - Analyzes expert performance and weight evolution

2. Online Learning with Expert Advice (Unsupervised) 
   - Implementation in `q1_cloud.ipynb`
   - Uses cloud dataset to compare different clustering algorithms
   - Implements static and fixed-share weight updates
   - Analyzes clustering performance and weight evolution

3. Loss Functions and Clustering Analysis
   - Implementation in `q4.ipynb`
   - Visualizes and compares different loss functions (squared, logistic, hinge, 0/1)
   - Demonstrates k-means clustering with good vs poor initialization
   - Uses synthetic and Iris datasets

## Setup

1. Create a Python environment using the provided `requirements.txt`:
```bash
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
pip install -r requirements.txt
```

2. Launch Jupyter notebook:
```bash
jupyter notebook
```

## Key Files

- `q1_spambase.ipynb`: Supervised online learning implementation using the Spambase dataset
- `q1_cloud.ipynb`: Unsupervised online learning implementation using the Cloud dataset
- `q4.ipynb`: Loss functions visualization and clustering analysis
- `requirements.txt`: Required Python packages and their versions

## Implementation Details

### Online Learning (q1_spambase.ipynb)
- Implements 6 expert classifiers: LogisticRegression, NaiveBayes, RandomForest, SVM, MLP, and GradientBoosting
- Compares static weights vs fixed-share weight updates
- Tests different α values (0.01, 0.1, 0.3) for fixed-share updates
- Visualizes expert weights and cumulative loss over time

### Online Learning (q1_cloud.ipynb)
- Implements 6 clustering experts: KMeans and GMM with k=2,3,4
- Compares static weights vs fixed-share weight updates
- Tests different α values (0.01, 0.1, 0.3) for fixed-share updates
- Visualizes clustering results and expert weight evolution

### Loss Functions & Clustering (q4.ipynb)
- Visualizes and compares key loss functions:
  - Squared loss
  - Logistic loss
  - Hinge loss
  - 0/1 loss
- Demonstrates k-means clustering sensitivity to initialization:
  - Good initialization with well-separated centers
  - Poor initialization with overlapping centers
- Uses both synthetic Gaussian clusters and the Iris dataset

## Running the Code

1. Each notebook can be run independently
2. Code cells should be executed sequentially
3. Markdown cells provide detailed explanations of the implementation and analysis
4. Visualizations include:
   - Loss function plots
   - Expert weight evolution
   - Clustering results
   - Performance comparisons

## Dependencies

Key packages required:
- numpy
- matplotlib 
- scikit-learn
- pandas
- ucimlrepo

Full dependencies are listed in `requirements.txt`.

## Notes for Grading

- The implementation follows the homework specifications for online learning algorithms
- Different α values are tested to demonstrate the impact on expert weight updates
- Visualizations are included to help understand algorithm behavior
- Analysis of results is provided in markdown cells
- Code is documented and organized for clarity