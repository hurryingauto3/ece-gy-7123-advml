%% Main Comparison Script for Optimization Algorithms
clear; clc; close all;

% --- Configuration ---
global_lambda = 0.1;
global_tol = 1e-5;
global_maxIter = 500;
global_stepSizes = [1e-6, 1e-5, 1e-4, 1e-3]; % Shared step sizes

results = struct(); % Store BEST results for final comparison
all_run_data = struct(); % Store data for ALL runs for iteration plots

% --- Data Loading & Preprocessing ---
% ... (Keep this section as before) ...
fprintf('Loading data...\n');
load('data1.mat');
if ~exist('TrainingX','var') || ~exist('TrainingY','var') || ...
   ~exist('TestX','var') || ~exist('TestY','var')
    error('Data file missing variables.');
end
TrainingY = TrainingY(:);
TestY = TestY(:);
N_full = size(TrainingX, 1);
fprintf('Data loaded. N_train=%d, N_test=%d\n', N_full, size(TestX, 1));

% --- Kernel Calculation ---
fprintf('Calculating sigma^2 and Full Kernel Matrix K_train...\n');
D_train_full_sq = pdist2(TrainingX, TrainingX, 'squaredeuclidean');
sigma_k_sq = sum(D_train_full_sq(:)) / (N_full^2);
K_train = exp(-D_train_full_sq / (2 * sigma_k_sq));
fprintf('Sigma^2 = %g. K_train calculated.\n', sigma_k_sq);
clear D_train_full_sq;

% Precompute Full Test Kernel
fprintf('Calculating Full Test Kernel Matrix K_test_full...\n');
D_test_full = pdist2(TestX, TrainingX, 'squaredeuclidean');
K_test_full = exp(-D_test_full / (2 * sigma_k_sq));
clear D_test_full;
fprintf('K_test_full calculated.\n');

GD_run = true;
SGD_run = true;
BFGS_run = true;
LBFGS_run = true;

% --- Experiment Setup ---
fprintf('Setting up experiments...\n');

% --- Gradient Descent (GD) ---
if GD_run == true
    % --- Experiment A: Gradient Descent (GD) ---
    fprintf('\n===== Running GD Experiment =====\n');
    gd_options.tol = global_tol;
    gd_options.maxIter = global_maxIter;
    num_gd_runs = length(global_stepSizes);
    gd_accuracies = zeros(num_gd_runs, 1);
    all_run_data.GD.histories = cell(num_gd_runs, 1); % Store all histories
    all_run_data.GD.params = cell(num_gd_runs, 1);   % Store parameters for legend
    best_gd_acc = -1;

    for i = 1:num_gd_runs
        eta = global_stepSizes(i);
        param_str = sprintf('eta=%g', eta); % String for legend
        fprintf('--- GD with %s ---\n', param_str);

        gd_options.eta = eta;
        [omega_gd, history_gd] = gd_optimizer(K_train, TrainingY, global_lambda, gd_options);
        [accuracy_gd, predictions_gd] = evaluate_model(omega_gd, K_test_full, TestY);

        gd_accuracies(i) = accuracy_gd;
        all_run_data.GD.histories{i} = history_gd; % Store history
        all_run_data.GD.params{i} = param_str;   % Store param string

        fprintf('GD (%s) Final Accuracy: %.2f%%\n', param_str, accuracy_gd*100);

        % Keep track of best result for final comparison plots
        if accuracy_gd > best_gd_acc
            best_gd_acc = accuracy_gd;
            results.GD.omega = omega_gd;
            results.GD.history = history_gd; % Store best history here too
            results.GD.accuracy = accuracy_gd;
            results.GD.predictions = predictions_gd;
            results.GD.best_params_str = param_str;
        end
    end
else
    fprintf('Skipping GD Experiment.\n');
end

% --- Stochastic Gradient Descent (SGD) ---
if SGD_run == true
    % --- Experiment B: Stochastic Gradient Descent (SGD) ---
    fprintf('\n===== Running SGD Experiment =====\n');
    sgd_options.tol = global_tol;
    sgd_options.maxIter = global_maxIter;
    sgd_batchSizes = [1, 100];
    num_sgd_ss = length(global_stepSizes);
    num_sgd_bs = length(sgd_batchSizes);
    num_sgd_runs = num_sgd_ss * num_sgd_bs;
    sgd_accuracies = zeros(num_sgd_ss, num_sgd_bs);
    all_run_data.SGD.histories = cell(num_sgd_runs, 1);
    all_run_data.SGD.params = cell(num_sgd_runs, 1);
    best_sgd_acc = -1;
    run_idx_sgd = 1;

    for j = 1:num_sgd_bs
        bs = sgd_batchSizes(j);
        sgd_options.batchSize = bs;
        for i = 1:num_sgd_ss
            eta = global_stepSizes(i);
            param_str = sprintf('eta=%g, bs=%d', eta, bs);
            fprintf('--- SGD with %s ---\n', param_str);

            sgd_options.eta = eta;
            [omega_sgd, history_sgd] = sgd_optimizer(K_train, TrainingY, global_lambda, sgd_options);
            [accuracy_sgd, predictions_sgd] = evaluate_model(omega_sgd, K_test_full, TestY);

            sgd_accuracies(i, j) = accuracy_sgd;
            all_run_data.SGD.histories{run_idx_sgd} = history_sgd;
            all_run_data.SGD.params{run_idx_sgd} = param_str;
            run_idx_sgd = run_idx_sgd + 1;

            fprintf('SGD (%s) Final Accuracy: %.2f%%\n', param_str, accuracy_sgd*100);

            % Keep track of best result
            if accuracy_sgd > best_sgd_acc
                best_sgd_acc = accuracy_sgd;
                results.SGD.omega = omega_sgd;
                results.SGD.history = history_sgd;
                results.SGD.accuracy = accuracy_sgd;
                results.SGD.predictions = predictions_sgd;
                results.SGD.best_params_str = param_str;
            end
        end
    end
else
    fprintf('Skipping SGD Experiment.\n');
end

% --- Prepare Reduced Data for BFGS/L-BFGS ---
% ... (Keep this section as before, including calculating K_reduced and K_test_reduced) ...
fprintf('\n--- Preparing Reduced Data for (L)BFGS ---\n');
N_reduced = 4000;
N_per_class = N_reduced / 2;
fprintf('Subsampling training data (%d per class)...\n', N_per_class);
idx_pos = find(TrainingY == 1);
idx_neg = find(TrainingY == -1);
if length(idx_pos) < N_per_class || length(idx_neg) < N_per_class
    error('Not enough samples in each class for the required subsample size.');
end
rand_idx_pos = idx_pos(randperm(length(idx_pos), N_per_class));
rand_idx_neg = idx_neg(randperm(length(idx_neg), N_per_class));
idx_reduced = sort([rand_idx_pos; rand_idx_neg]);
X_reduced = TrainingX(idx_reduced, :);
y_reduced = TrainingY(idx_reduced);
fprintf('Subsampled data size: %d x %d\n', size(X_reduced, 1), size(X_reduced, 2));

fprintf('Calculating reduced kernel matrix (K_reduced)...\n');
D_reduced_sq = pdist2(X_reduced, X_reduced, 'squaredeuclidean');
K_reduced = exp(-D_reduced_sq / (2 * sigma_k_sq));
fprintf('Reduced kernel matrix size: %d x %d\n', size(K_reduced, 1), size(K_reduced, 2));
clear D_reduced_sq;

fprintf('Calculating reduced test kernel matrix (K_test_reduced)...\n');
D_test_reduced_sq = pdist2(TestX, X_reduced, 'squaredeuclidean');
K_test_reduced = exp(-D_test_reduced_sq / (2 * sigma_k_sq));
clear D_test_reduced_sq;

% --- Experiment C: BFGS ---
if BFGS_run == true
    % --- Experiment C: BFGS ---
    fprintf('\n===== Running BFGS Experiment =====\n');
    % Note: BFGS has no hyperparameters like step size to loop through here
    bfgs_options.tol = global_tol;
    bfgs_options.maxIter = global_maxIter;
    bfgs_options.alpha0 = 1.0;
    bfgs_options.c1 = 1e-4;
    bfgs_options.rho = 0.5;
    param_str_bfgs = 'BFGS'; % Simple label

    [omega_bfgs, history_bfgs] = bfgs_optimizer(K_reduced, y_reduced, global_lambda, bfgs_options);
    [accuracy_bfgs, predictions_bfgs] = evaluate_model(omega_bfgs, K_test_reduced, TestY);
    fprintf('BFGS Final Accuracy: %.2f%%\n', accuracy_bfgs*100);

    % Store results (only one run for BFGS in this setup)
    all_run_data.BFGS.histories = {history_bfgs};
    all_run_data.BFGS.params = {param_str_bfgs};
    results.BFGS.omega = omega_bfgs;
    results.BFGS.history = history_bfgs;
    results.BFGS.accuracy = accuracy_bfgs;
    results.BFGS.predictions = predictions_bfgs;
    results.BFGS.best_params_str = param_str_bfgs;
else
    fprintf('Skipping BFGS Experiment.\n');
end

% --- Experiment D: L-BFGS ---
if LBFGS_run == true
    fprintf('\n===== Running L-BFGS Experiment =====\n');
    lbfgs_options = bfgs_options; % Inherit line search params etc.
    lbfgs_memories = [5, 10, 20, 40, 80]; % Experiment with different memory sizes
    num_lbfgs_runs = length(lbfgs_memories);
    lbfgs_accuracies = zeros(num_lbfgs_runs, 1);
    all_run_data.LBFGS.histories = cell(num_lbfgs_runs, 1);
    all_run_data.LBFGS.params = cell(num_lbfgs_runs, 1);
    best_lbfgs_acc = -1;

    for i = 1:num_lbfgs_runs
        m = lbfgs_memories(i);
        param_str = sprintf('m=%d', m);
        fprintf('--- L-BFGS with %s ---\n', param_str);

        lbfgs_options.m = m;
        [omega_lbfgs, history_lbfgs] = lbfgs_optimizer(K_reduced, y_reduced, global_lambda, lbfgs_options);
        [accuracy_lbfgs, predictions_lbfgs] = evaluate_model(omega_lbfgs, K_test_reduced, TestY);

        lbfgs_accuracies(i) = accuracy_lbfgs;

        all_run_data.LBFGS.histories{i} = history_lbfgs;
        all_run_data.LBFGS.params{i} = param_str;

        fprintf('L-BFGS (%s) Final Accuracy: %.2f%%\n', param_str, accuracy_lbfgs*100);

        if accuracy_lbfgs > best_lbfgs_acc
                best_lbfgs_acc = accuracy_lbfgs;
                results.LBFGS.omega = omega_lbfgs;
                results.LBFGS.history = history_lbfgs;
                results.LBFGS.accuracy = accuracy_lbfgs;
                results.LBFGS.predictions = predictions_lbfgs;
                results.LBFGS.best_params_str = param_str; % Store best 'm' info
        end
    end
else
    fprintf('Skipping L-BFGS Experiment.\n');
end

% --- Iteration Plots ---
fprintf('\n===== Generating Iteration Plots =====\n');
% Plot GD Accuracy vs Step Size
if GD_run == true
    plot_accuracy_hyperparam(global_stepSizes, 1, gd_accuracies, ...
        'Step Size (\eta)', '', 'GD Accuracy vs. Step Size');
    plot_iteration_convergence(all_run_data.GD.histories, all_run_data.GD.params,...
    'GD Cost vs Iteration', 'Iteration', 'Cost J(omega)', 'cost', true);
    plot_iteration_convergence(all_run_data.GD.histories, all_run_data.GD.params,...
    'GD Grad Norm vs Iteration', 'Iteration', '||Grad||', 'grad_norm', true);
else
    fprintf('Skipping GD Iteration Plot.\n');
end

% Plot SGD Accuracy vs Step Size for different Batch Sizes
if SGD_run == true
    plot_accuracy_hyperparam(global_stepSizes, sgd_batchSizes, sgd_accuracies, ...
    'Step Size (\eta)', 'Batch Size', 'SGD Accuracy vs. Step Size');
    plot_iteration_convergence(all_run_data.SGD.histories, all_run_data.SGD.params,...
        'SGD Cost vs Iteration', 'Iteration', 'Cost J(omega)', 'cost', true);
    plot_iteration_convergence(all_run_data.SGD.histories, all_run_data.SGD.params,...
    'SGD Grad Norm vs Iterations', 'Iterations', '||Grad||', 'grad_norm', true);
else
    fprintf('Skipping SGD Iteration Plot.\n');
end


if BFGS_run == true
    plot_iteration_convergence(all_run_data.BFGS.histories, all_run_data.BFGS.params,...
    'BFGS Cost vs Iteration', 'Iteration', 'Cost J(omega)', 'cost', true);
    plot_iteration_convergence(all_run_data.BFGS.histories, all_run_data.BFGS.params,...
    'BFGS Grad Norm vs Iteration', 'Iteration', '||Grad||', 'grad_norm', true);

else
    fprintf('Skipping BFGS Iteration Plot.\n');
end

if LBFGS_run == true
    plot_accuracy_hyperparam(lbfgs_memories, 1, lbfgs_accuracies, ...
        'Memory Size ()', '', 'LBFGS Accuracy vs. Memory Size');

    plot_iteration_convergence(all_run_data.LBFGS.histories, all_run_data.LBFGS.params,...
        'L-BFGS Cost vs Iteration', 'Iteration', 'Cost J(omega)', 'cost', true);
    
        plot_iteration_convergence(all_run_data.LBFGS.histories, all_run_data.LBFGS.params,...
    'L-BFGS Grad Norm vs Iteration', 'Iteration', '||Grad||', 'grad_norm', true);

else
    fprintf('Skipping L-BFGS Iteration Plot.\n');
end

% --- Final Comparison Plots ---
% ... (Keep plot_convergence and plot_pca_predictions calls using the 'results' struct as before) ...
fprintf('\n===== Generating Final Comparison Plots =====\n');
plot_convergence(results, ' (Best Runs)');
plot_pca_predictions(TestX, results, TestY, ' (Best Runs)');

% --- Display Final Accuracies ---
% ... (Keep this section as before) ...
fprintf('\n===== Final Accuracies (Best Runs) =====\n');
if isfield(results, 'GD'), fprintf('GD (%s): %.2f%%\n', results.GD.best_params_str, results.GD.accuracy*100); end
if isfield(results, 'SGD'), fprintf('SGD (%s): %.2f%%\n', results.SGD.best_params_str, results.SGD.accuracy*100); end
if isfield(results, 'BFGS'), fprintf('BFGS (%s): %.2f%%\n', results.BFGS.best_params_str, results.BFGS.accuracy*100); end
if isfield(results, 'LBFGS'), fprintf('L-BFGS (%s): %.2f%%\n', results.LBFGS.best_params_str, results.LBFGS.accuracy*100); end
fprintf('======================================\n');