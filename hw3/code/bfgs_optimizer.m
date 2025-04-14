function [omega, history] = bfgs_optimizer(K_reduced, y_reduced, lambda, options)
    % Performs manual BFGS optimization for Kernel Logistic Regression.
    % Inputs:
    %   K_reduced - Reduced kernel matrix (N_reduced x N_reduced)
    %   y_reduced - Labels for the reduced dataset (N_reduced x 1)
    %   lambda    - Regularization parameter
    %   options   - Struct with: .maxIter, .tol, .alpha0, .c1, .rho
    % Outputs:
    %   omega   - Optimized parameter vector (N_reduced x 1)
    %   history - Struct with: .cost, .grad_norm, .time
    
    N_reduced = size(K_reduced, 1);

    omega = zeros(N_reduced, 1);
    H = eye(N_reduced);
    history.cost = zeros(options.maxIter + 1, 1);
    history.grad_norm = zeros(options.maxIter + 1, 1);
    history.time = zeros(options.maxIter + 1, 1);
    iter_start_time = tic;

    % Initial cost and gradient
    [current_cost, current_grad] = kernelLogisticCostGrad(omega, K_reduced, y_reduced, lambda);
    grad_norm = norm(current_grad);
    history.cost(1) = current_cost;
    history.grad_norm(1) = grad_norm;
    history.time(1) = toc(iter_start_time);
    fprintf('BFGS Iter 0: Cost = %g, Grad Norm = %g\n', current_cost, grad_norm);

    % BFGS optimization loop
    k = 0;
    while k < options.maxIter && grad_norm > options.tol
        p = -H * current_grad;
        directional_deriv = current_grad' * p;

        if directional_deriv >= 0
            fprintf('Warning: BFGS search direction not descent (g''*p = %g) at iter %d. Resetting H.\n', directional_deriv, k+1);
            H = eye(N_reduced);
            p = -H * current_grad;
            directional_deriv = current_grad' * p;
            if directional_deriv >= 0, fprintf('Error: Cannot find descent direction. Stopping.\n'); break; end
        end

        alpha = options.alpha0;
        cost_threshold = current_cost + options.c1 * alpha * directional_deriv;
        omega_new_trial = omega + alpha * p;
        [cost_new_trial, ~] = kernelLogisticCostGrad(omega_new_trial, K_reduced, y_reduced, lambda);

        line_search_iters = 0;
        % Line search to find suitable alpha
        while cost_new_trial > cost_threshold
            alpha = options.rho * alpha;
            if alpha < 1e-12, fprintf('Warning: Line search alpha too small.\n'); alpha = alpha/options.rho; break; end % Revert alpha
            omega_new_trial = omega + alpha * p;
            [cost_new_trial, ~] = kernelLogisticCostGrad(omega_new_trial, K_reduced, y_reduced, lambda);
            cost_threshold = current_cost + options.c1 * alpha * directional_deriv;
                line_search_iters = line_search_iters + 1;
                if line_search_iters > 50, fprintf('Warning: Line search max iters. Using current alpha.\n'); break; end
        end
        alpha_k = alpha;
        
        % Update omega and compute new cost and gradient
        omega_next = omega + alpha_k * p;
        [cost_next, grad_next] = kernelLogisticCostGrad(omega_next, K_reduced, y_reduced, lambda);
        delta = omega_next - omega;
        gamma = grad_next - current_grad;
        rho_denom = gamma' * delta;
        
        % BFGS update
        if rho_denom > 1e-10
            rho_k = 1 / rho_denom;
            I_mat = eye(N_reduced);
            term1 = (I_mat - rho_k * delta * gamma');
            term2 = (I_mat - rho_k * gamma * delta');
            H = term1 * H * term2 + rho_k * (delta * delta');
        elseif k > 0
                fprintf('Warning: BFGS Curvature condition <= 0 at iter %d. Skipping H update.\n', k+1);
        end

        omega = omega_next;
        current_grad = grad_next;
        current_cost = cost_next;
        grad_norm = norm(current_grad);
        k = k + 1;

        history.cost(k+1) = current_cost;
        history.grad_norm(k+1) = grad_norm;
        history.time(k+1) = toc(iter_start_time);

        if mod(k, 10) == 0 || k == options.maxIter || grad_norm <= options.tol
            fprintf('BFGS Iter %d: Cost = %g, Grad Norm = %g, Step Size = %g\n', k, current_cost, grad_norm, alpha_k);
        end
    end

    history.cost = history.cost(1:k+1);
    history.grad_norm = history.grad_norm(1:k+1);
    history.time = history.time(1:k+1);
    history.total_time = toc(iter_start_time);
    fprintf('Manual BFGS finished after %d iterations.\n', k);
end