# Advanced Machine Learning - Homework 3: Optimization Algorithm Comparison

This repository contains MATLAB code for comparing different optimization algorithms (Gradient Descent, Stochastic Gradient Descent, BFGS, and L-BFGS) for Kernel Logistic Regression.

## Project Structure

- `code/`: Contains all the MATLAB source code and the data file.
    - `main_comparison.m`: The main script to run all experiments, tune hyperparameters, and generate comparison plots.
    - `kernelLogisticCostGrad.m`: Calculates the cost and gradient for the Kernel Logistic Regression objective function. Used by all optimizers.
    - `evaluate_model.m`: Evaluates the trained model's accuracy on the test set and returns predictions.
    - `gd_optimizer.m`: Implements the Gradient Descent (GD) optimizer.
    - `sgd_optimizer.m`: Implements the Stochastic Gradient Descent (SGD) optimizer.
    - `bfgs_optimizer.m`: Implements the Broyden–Fletcher–Goldfarb–Shanno (BFGS) optimizer.
    - `lbfgs_optimizer.m`: Implements the Limited-memory BFGS (L-BFGS) optimizer.
    - `plot_convergence.m`: Generates plots comparing the cost and gradient norm convergence over time for the best runs of each optimizer.
    - `plot_pca_predictions.m`: Generates PCA plots visualizing the decision boundaries and predictions for the best runs of each optimizer.
    - `plot_accuracy_hyperparam.m`: Generates plots showing the test accuracy as a function of hyperparameters (e.g., step size, batch size, memory size).
    - `data1.mat`: The dataset file containing `TrainingX`, `TrainingY`, `TestX`, `TestY`.
- `report/`: Contains the LaTeX source and generated PDF report (`hw3_ah7072.pdf`) discussing the implementation and results.

## Requirements

- MATLAB (tested on R2024a or later)

## Usage

1.  Ensure you have MATLAB installed.
2.  Navigate to the `code/` directory in MATLAB.
3.  Make sure the `data1.mat` file is present in the `code/` directory.
4.  Run the main script from the MATLAB Command Window:
    ```matlab
    main
    ```
5.  The script will execute the experiments for each optimizer, tune hyperparameters (for GD, SGD, L-BFGS), store results, and generate comparison plots.

## Outputs

The `main.m` script will:
- Print progress and final accuracy results for each optimizer to the MATLAB Command Window.
- Generate and save several plots in the `code/` directory :
    - Accuracy vs. Hyperparameter plots (e.g., `gd_accuracy_vs_step_size.png`, `sgd_accuracy_vs_step_size.png`).
    - Convergence plots showing cost and gradient norm vs. time (`convergence_comparison_Best_Runs.png`).
    - PCA prediction plots comparing optimizer results (`pca_predictions_comparison_Best_Runs.png`).

