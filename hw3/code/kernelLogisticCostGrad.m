function [cost, grad] = kernelLogisticCostGrad(omega, K, y, lambda)
    % Calculates the cost and gradient for Kernel Logistic Regression.
    % Inputs:
    %   omega  - Current parameter vector (N x 1 or N_reduced x 1)
    %   K      - Kernel matrix (N x N or N_reduced x N_reduced)
    %   y      - Labels vector (N x 1 or N_reduced x 1)
    %   lambda - Regularization parameter
    % Outputs:
    %   cost   - Scalar cost value J(omega)
    %   grad   - Gradient vector (same size as omega)
    
        N = size(K, 1); % Number of samples (either N or N_reduced)
    
        % --- Calculate Cost ---
        a = K * omega; % (N x 1)
        z = y .* a;
    
        % Stable Log Likelihood Calculation using log-sum-exp trick
        % J = sum(log(1+exp(-z_i*y_i))) + lambda*omega'*omega where z_i = k_i'*omega
        % Let v = -y .* a = -z
        v = -z;
        % log(1+exp(x)) = max(x,0) + log(1+exp(-abs(x)))
        log_one_plus_exp_neg_z = max(v, 0) + log(1 + exp(-abs(v)));
        log_likelihood = sum(log_one_plus_exp_neg_z); % This is -sum(log(sigma(z)))
    
        regularization = lambda * (omega' * omega);
        cost = log_likelihood + regularization;
    
        % --- Calculate Gradient ---
        % sigma(z) = 1 / (1 + exp(-z))
        sigma_z = 1 ./ (1 + exp(-z)); % Need sigma(z) for gradient
        % Assuming K is symmetric (true for RBF)
        grad_ll_part = K * ((sigma_z - 1) .* y);
        grad_reg_part = 2 * lambda * omega;
        grad = grad_ll_part + grad_reg_part;
    
    end