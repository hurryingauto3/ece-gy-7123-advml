function omega = gd(K, y, lambda, eta, tol, maxIter)
%GRADIENTDESCENT Performs gradient descent for Kernel Logistic Regression.
%   omega = gradientDescent(K, y, lambda, eta, tol, maxIter) returns the 
%   optimized parameter vector omega given:
%       K       - Kernel matrix (N x N) where each column represents k_i.
%       y       - Labels vector (N x 1).
%       lambda  - Regularization parameter.
%       eta     - Learning rate.
%       tol     - Convergence tolerance (based on the gradient norm).
%       maxIter - Maximum number of iterations.
%
%   The function minimizes the objective:
%       J(omega) = -sum_{i=1}^{N} log( sigma(y_i*(omega^T * k_i)) ) + lambda * omega' * omega,
%   where sigma(z) = 1/(1+exp(-z)).

N = size(K, 1);
omega = zeros(N, 1);  % Initialize omega

for t = 1:maxIter
    % Compute the decision values: a = K * omega, where each a(i) = omega^T * k_i.
    a = K * omega;
    
    % Compute z = y .* a (element-wise multiplication)
    z = y .* a;
    
    % Compute the sigmoid function sigma(z) = 1/(1 + exp(-z))
    sigma_z = 1 ./ (1 + exp(-z));
    
    % Compute the gradient:
    % g = sum_i (sigma(z_i) - 1) * y_i * k_i + 2 * lambda * omega
    % Vectorized as:
    g = K' * ((sigma_z - 1) .* y) + 2 * lambda * omega;
    
    % Check convergence based on the norm of the gradient
    if norm(g) < tol
        fprintf('Converged at iteration %d\n', t);
        break;
    end
    
    % Update omega
    omega = omega - eta * g;
end

end