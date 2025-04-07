%% main.m
clear; clc;

%% Load Data
load('data1.mat'); % Expects TrainingX, TrainingY, TestX, TestY
if ~exist('TrainingX','var') || ~exist('TrainingY','var') || ...
        ~exist('TestX','var') || ~exist('TestY','var')
    error('Data file must contain TrainingX, TrainingY, TestX, and TestY.');
end

% Ensure labels are column vectors
TrainingY = TrainingY(:);
TestY = TestY(:);

%% Compute Kernel Matrix for Training Data (RBF Kernel)
N = size(TrainingX, 1);
D_train = pdist2(TrainingX, TrainingX, 'euclidean').^2;
sigma_k_sq = sum(D_train(:)) / (N^2);
K_train = exp(-D_train/(2*sigma_k_sq));

%% Set Optimizer Type and Hyperparameters
optimizerType = 'GD';  % Options: 'GD', 'SGD', 'BFGS', 'LBFGS'
lambda = 0.1;          % Regularization parameter
eta = 0.01;            % Learning rate (for GD and SGD)
tol = 1e-6;            % Convergence tolerance
maxIter = 1000;        % Maximum iterations
memorySize = 10;       % Memory size for L-BFGS (if used)

%% Train Kernel Logistic Regression Model
switch optimizerType
    case 'GD'
        fprintf('Using Gradient Descent...\n');
        omega = gd(K_train, TrainingY, lambda, eta, tol, maxIter);
    % Uncomment and implement these cases if needed:
    % case 'SGD'
    %     fprintf('Using Stochastic Gradient Descent...\n');
    %     omega = sgd(K_train, TrainingY, lambda, eta, tol, maxIter);
    % case 'BFGS'
    %     fprintf('Using BFGS...\n');
    %     omega = bfgsOptimizer(K_train, TrainingY, lambda, tol, maxIter);
    % case 'LBFGS'
    %     fprintf('Using L-BFGS...\n');
    %     omega = lbfgsOptimizer(K_train, TrainingY, lambda, tol, maxIter, memorySize);
    otherwise
        error('Unknown optimizer type.');
end

%% Evaluate Model on Test Data
% Compute kernel matrix between test and training data
D_test = pdist2(TestX, TrainingX, 'euclidean').^2;
K_test = exp(-D_test/(2*sigma_k_sq));

% Compute decision values for test data
a_test = K_test * omega;

% Compute predicted probabilities using the sigmoid function
p_test = 1 ./ (1 + exp(-a_test));

% Convert probabilities to class labels (threshold at 0.5)
predictedLabels = ones(size(p_test));
predictedLabels(p_test < 0.5) = -1;

% Compute and display accuracy
accuracy = mean(predictedLabels == TestY);
fprintf('Test Accuracy using %s: %.2f%%\n', optimizerType, accuracy * 100);

%% Plot PCA Projections for Predicted vs. True Labels
% plotPCAPredictions(TestX, predictedLabels, TestY);

%% Plot Accuracies vs. Various Step Sizes (Gradient Descent)
stepSizes = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5];
plotAccuracyVsStepSizes(TrainingX, TrainingY, TestX, TestY, sigma_k_sq, lambda, tol, maxIter, stepSizes);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Subfunction: plotPCAPredictions
function plotPCAPredictions(TestX, predictedLabels, trueLabels)
    % Perform PCA on TestX
    [coeff, score, ~, ~, explained, ~] = pca(TestX);
    
    % Create figure with subplots for predicted and true labels
    figure;
    subplot(1,2,1);
    gscatter(score(:,1), score(:,2), predictedLabels);
    title('PCA of Test Data (Predicted Labels)');
    xlabel('Principal Component 1');
    ylabel('Principal Component 2');
    
    subplot(1,2,2);
    gscatter(score(:,1), score(:,2), trueLabels);
    title('PCA of Test Data (True Labels)');
    xlabel('Principal Component 1');
    ylabel('Principal Component 2');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Subfunction: plotAccuracyVsStepSizes
function plotAccuracyVsStepSizes(TrainingX, TrainingY, TestX, TestY, sigma_k_sq, lambda, tol, maxIter, stepSizes)
    accuracies = zeros(size(stepSizes));
    N = size(TrainingX, 1);
    
    % Compute kernel matrix for training data
    D_train = pdist2(TrainingX, TrainingX, 'euclidean').^2;
    K_train = exp(-D_train/(2*sigma_k_sq));
    
    for i = 1:length(stepSizes)
        
        currentEta = stepSizes(i);
        % Train using Gradient Descent for current step size
        omega_temp = gd(K_train, TrainingY, lambda, currentEta, tol, maxIter);
        
        % Evaluate on test data
        D_test = pdist2(TestX, TrainingX, 'euclidean').^2;
        K_test = exp(-D_test/(2*sigma_k_sq));
        a_test = K_test * omega_temp;
        p_test = 1 ./ (1 + exp(-a_test));
        predictedLabels_temp = ones(size(p_test));
        predictedLabels_temp(p_test < 0.5) = -1;
        
        % Compute accuracy
        accuracies(i) = mean(predictedLabels_temp == TestY);
        plotPCAPredictions(TestX, predictedLabels_temp, TestY);
    end
    
    % Plot test accuracy vs. step sizes
    figure;
    plot(stepSizes, accuracies*100, '-o', 'LineWidth', 2);
    xlabel('Step Size (\eta)');
    ylabel('Test Accuracy (%)');
    title('Test Accuracy vs. Step Size (Gradient Descent)');
    grid on;
end