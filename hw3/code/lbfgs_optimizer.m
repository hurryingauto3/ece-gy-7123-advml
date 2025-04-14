function [omega, history] = lbfgs_optimizer(K_reduced, y_reduced, lambda, options)
    % Performs L-BFGS optimization for Kernel Logistic Regression.
    % Inputs:
    %   K_reduced - Reduced kernel matrix (N_reduced x N_reduced)
    %   y_reduced - Labels for the reduced dataset (N_reduced x 1)
    %   lambda    - Regularization parameter
    %   options   - Struct with: .maxIter, .tol, .alpha0, .c1, .rho, .m (memory)
    % Outputs:
    %   omega   - Optimized parameter vector (N_reduced x 1)
    %   history - Struct with: .cost, .grad_norm, .time

    N_reduced = size(K_reduced, 1);
    m = options.m; % L-BFGS memory parameter

    omega = zeros(N_reduced, 1);
    history.cost = zeros(options.maxIter + 1, 1);
    history.grad_norm = zeros(options.maxIter + 1, 1);
    history.time = zeros(options.maxIter + 1, 1);
    iter_start_time = tic;

    % Store previous m vectors for s and y - Use direct indexing 1..m
    s_history = zeros(N_reduced, m);
    y_history = zeros(N_reduced, m);
    rho_history = zeros(1, m); % Stores 1 / (y_k' * s_k)
    history_count = 0; % Number of pairs currently stored

    fprintf('Starting Manual L-BFGS (Memory m=%d, Max Iter: %d, Tol: %g)...\n', m, options.maxIter, options.tol);

    [current_cost, current_grad] = kernelLogisticCostGrad(omega, K_reduced, y_reduced, lambda);
    grad_norm = norm(current_grad);
    history.cost(1) = current_cost;
    history.grad_norm(1) = grad_norm;
    history.time(1) = toc(iter_start_time);
    fprintf('L-BFGS Iter 0: Cost = %g, Grad Norm = %g\n', current_cost, grad_norm);

    k = 0;
    while k < options.maxIter && grad_norm > options.tol

        % --- 1. Compute Search Direction (L-BFGS two-loop recursion) ---
        q = current_grad;
        alpha_hist = zeros(1, history_count); % Store alpha_i values needed later

        % First loop (backward pass, using stored pairs from newest to oldest)
        for i = history_count:-1:1
             alpha_hist(i) = rho_history(i) * (s_history(:, i)' * q);
             q = q - alpha_hist(i) * y_history(:, i);
        end

        % Scaling initial Hessian approximation H_k^0
        if history_count > 0 % Use heuristic only if we have stored pairs
            y_k_prev = y_history(:, history_count); % Use the MOST RECENT pair
            s_k_prev = s_history(:, history_count);
            ys = y_k_prev' * s_k_prev;
            ss = s_k_prev' * s_k_prev;
             if ys > 1e-10 && ss > 1e-10 % Check both numerator and denominator
                 Hk0_scale_factor = ys / ss; % Nocedal&Wright notation gamma_k
             else
                 Hk0_scale_factor = 1.0;
             end
        else
            Hk0_scale_factor = 1.0; % H0 = I
        end
        z = (1.0 / Hk0_scale_factor) * q; % Apply H0 approximation (H0 = I/gamma_k)

        % Second loop (forward pass, using stored pairs from oldest to newest)
        for i = 1:history_count
             beta = rho_history(i) * (y_history(:, i)' * z);
             z = z + s_history(:, i) * (alpha_hist(i) - beta);
        end
        p = -z; % Final search direction

        % --- 2. Line Search (Backtracking - same as BFGS) ---
         directional_deriv = current_grad' * p;
         if directional_deriv >= 0 % Check if descent direction
            fprintf('Warning: L-BFGS search direction not descent (g''*p = %g) at iter %d. Using -gradient.\n', directional_deriv, k+1);
            p = -current_grad; % Fallback to gradient descent direction
            directional_deriv = current_grad'*p; % Recalculate directional derivative
            if directional_deriv >= 0, fprintf('Error: Cannot find descent direction even with -g. Stopping.\n'); break; end
         end

         alpha = options.alpha0;
         cost_threshold = current_cost + options.c1 * alpha * directional_deriv;
         omega_new_trial = omega + alpha * p;
         [cost_new_trial, ~] = kernelLogisticCostGrad(omega_new_trial, K_reduced, y_reduced, lambda);

         line_search_iters = 0;
         while cost_new_trial > cost_threshold
             alpha = options.rho * alpha;
             if alpha < 1e-12 % Prevent tiny steps
                 fprintf('Warning: L-BFGS Line search alpha reached minimum (%g).\n', alpha);
                 % If alpha started at 1, options.rho=0.5, this takes ~40 steps.
                 % If it gets here, might be stuck. Let's allow using this small alpha.
                 break;
             end
             omega_new_trial = omega + alpha * p;
             [cost_new_trial, ~] = kernelLogisticCostGrad(omega_new_trial, K_reduced, y_reduced, lambda);
             cost_threshold = current_cost + options.c1 * alpha * directional_deriv;
             line_search_iters = line_search_iters + 1;
             if line_search_iters > 50 % Safety break
                  fprintf('Warning: L-BFGS Line search exceeded max iterations (%d). Using current alpha=%g.\n', line_search_iters, alpha);
                  break;
             end
         end
         alpha_k = alpha; % Use the found alpha (even if tiny)

        % --- 3. Update Parameters ---
        omega_next = omega + alpha_k * p;

        % --- 4. Compute New Gradient ---
        [cost_next, grad_next] = kernelLogisticCostGrad(omega_next, K_reduced, y_reduced, lambda);

        % --- 5. Compute delta and gamma & Update History Buffer ---
        delta = omega_next - omega;
        gamma = grad_next - current_grad;

        rho_denom = gamma' * delta;
         if rho_denom > 1e-10 % Store only if curvature condition holds
             if history_count < m % Buffer not full, append
                 history_count = history_count + 1;
                 s_history(:, history_count) = delta;
                 y_history(:, history_count) = gamma;
                 rho_history(history_count) = 1 / rho_denom;
             else % Buffer full, shift left and add to end
                 s_history(:, 1:m-1) = s_history(:, 2:m);
                 y_history(:, 1:m-1) = y_history(:, 2:m);
                 rho_history(1:m-1) = rho_history(2:m);
                 % Add new pair at the end (index m)
                 s_history(:, m) = delta;
                 y_history(:, m) = gamma;
                 rho_history(m) = 1 / rho_denom;
                 % history_count remains m
             end
         else
              if k > 0 % Avoid warning on first potential storage
                 fprintf('Warning: L-BFGS Curvature condition <= 0 at iter %d (rho_denom=%g). Discarding s,y pair.\n', k+1, rho_denom);
              end
         end

        % --- 6. Prepare for next iteration ---
        omega = omega_next;
        current_grad = grad_next;
        current_cost = cost_next;
        grad_norm = norm(current_grad);
        k = k + 1;

        history.cost(k+1) = current_cost;
        history.grad_norm(k+1) = grad_norm;
        history.time(k+1) = toc(iter_start_time);

        if mod(k, 10) == 0 || k == options.maxIter || grad_norm <= options.tol
            fprintf('L-BFGS Iter %d: Cost = %g, Grad Norm = %g, Step Size = %g\n', k, current_cost, grad_norm, alpha_k);
        end
    end % End of while loop

    history.cost = history.cost(1:k+1);
    history.grad_norm = history.grad_norm(1:k+1);
    history.time = history.time(1:k+1);
    history.total_time = toc(iter_start_time);
    fprintf('Manual L-BFGS finished after %d iterations.\n', k);
end