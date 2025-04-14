function [omega, history] = sgd_optimizer(K, y, lambda, options)
    % Performs Stochastic Gradient Descent for Kernel Logistic Regression.
    % Inputs:
    %   K       - Kernel matrix (N x N)
    %   y       - Label vector (N x 1)
    %   lambda  - Regularization parameter
    %   options - Struct with: .eta, .batchSize, .maxIter, .tol
    % Outputs:
    %   omega   - Optimized parameter vector (N x 1)
    %   history - Struct with: .cost (per epoch), .grad_norm (per epoch), .time (per epoch)
    
    N = length(y);
    omega = zeros(N, 1);

    history.cost = zeros(options.maxIter + 1, 1);
    history.grad_norm = zeros(options.maxIter + 1, 1); % Store norm of *full* gradient per epoch
    history.time = zeros(options.maxIter + 1, 1);
    epoch_start_time = tic;

    % Initial state (using full gradient for fair comparison)
    [cost, grad] = kernelLogisticCostGrad(omega, K, y, lambda);
    grad_norm = norm(grad);
    history.cost(1) = cost;
    history.grad_norm(1) = grad_norm;
    history.time(1) = toc(epoch_start_time);
        fprintf('SGD Epoch 0: Full Cost = %g, Full Grad Norm = %g\n', cost, grad_norm);

    converged = false;
    % Stochastic Gradient Descent loop
    for epoch = 1:options.maxIter
        randIdx = randperm(N);

        for startPos = 1:options.batchSize:N
            endPos = min(startPos + options.batchSize - 1, N);
            batchIdx = randIdx(startPos:endPos);
            current_batch_size = length(batchIdx);

            K_batch = K(:, batchIdx);
            y_batch = y(batchIdx);

            a_batch = K_batch' * omega;
            z_batch = y_batch .* a_batch;
            sigma_z_batch = 1 ./ (1 + exp(-z_batch));

            grad_loss_term = K_batch * ((sigma_z_batch - 1) .* y_batch);
            % Correctly scaled gradient for the batch
            grad_batch = (1 / current_batch_size) * grad_loss_term + 2 * lambda * omega;

            omega = omega - options.eta * grad_batch;
        end % End of mini-batch loop

        % Calculate and store full cost/gradient norm at the end of the epoch
        [cost, grad] = kernelLogisticCostGrad(omega, K, y, lambda);
        grad_norm = norm(grad);
        history.cost(epoch+1) = cost;
        history.grad_norm(epoch+1) = grad_norm;
        history.time(epoch+1) = toc(epoch_start_time);

        if mod(epoch, 50) == 0 || epoch == options.maxIter % Print progress
            fprintf('SGD Epoch %d: Full Cost = %g, Full Grad Norm = %g\n', epoch, cost, grad_norm);
        end

        % Check convergence based on full gradient norm
        if grad_norm <= options.tol
                fprintf('SGD Converged at epoch %d based on full gradient norm.\n', epoch);
                converged = true;
        end

        if converged
            break;
        end
    end % End of epoch loop

    % Trim history
    final_epochs = epoch + 1; % Number of entries stored
    history.cost = history.cost(1:final_epochs);
    history.grad_norm = history.grad_norm(1:final_epochs);
    history.time = history.time(1:final_epochs);
    history.total_time = toc(epoch_start_time);
end