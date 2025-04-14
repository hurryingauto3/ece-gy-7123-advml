function [omega, history] = gd_optimizer(K, y, lambda, options)
    % Performs Gradient Descent for Kernel Logistic Regression.
    % Inputs:
    %   K       - Kernel matrix (N x N)
    %   y       - Label vector (N x 1)
    %   lambda  - Regularization parameter
    %   options - Struct with: .eta, .tol, .maxIter
    % Outputs:
    %   omega   - Optimized parameter vector (N x 1)
    %   history - Struct with: .cost, .grad_norm, .time

    N = size(K, 1);
    omega = zeros(N, 1);

    history.cost = zeros(options.maxIter + 1, 1);
    history.grad_norm = zeros(options.maxIter + 1, 1);
    history.time = zeros(options.maxIter + 1, 1);
    iter_start_time = tic;

    % Initial state
    [cost, grad] = kernelLogisticCostGrad(omega, K, y, lambda);
    grad_norm = norm(grad);
    history.cost(1) = cost;
    history.grad_norm(1) = grad_norm;
    history.time(1) = toc(iter_start_time);
    fprintf('GD Iter 0: Cost = %g, Grad Norm = %g\n', cost, grad_norm);

    % Gradient Descent loop
    for k = 1:options.maxIter
        if grad_norm <= options.tol
            fprintf('GD Converged at iteration %d\n', k-1);
            break;
        end

        % Update omega
        omega = omega - options.eta * grad;

        % Calculate new state
        [cost, grad] = kernelLogisticCostGrad(omega, K, y, lambda);
        grad_norm = norm(grad);

        % Store history
        history.cost(k+1) = cost;
        history.grad_norm(k+1) = grad_norm;
        history.time(k+1) = toc(iter_start_time);

        if mod(k, 50) == 0 || k == options.maxIter % Print progress
                fprintf('GD Iter %d: Cost = %g, Grad Norm = %g\n', k, cost, grad_norm);
        end
    end

    % Trim history
    history.cost = history.cost(1:k);
    history.grad_norm = history.grad_norm(1:k);
    history.time = history.time(1:k);
    history.total_time = toc(iter_start_time);
end