function [accuracy, predictions] = evaluate_model(omega, K_test, TestY)
    % Evaluates the trained model on the test set.
    % Inputs:
    %   omega   - Trained parameter vector
    %   K_test  - Test kernel matrix (N_test x N or N_test x N_reduced)
    %   TestY   - True test labels
    % Outputs:
    %   accuracy    - Test accuracy (0 to 1)
    %   predictions - Predicted labels (-1 or 1) for the test set
    
        a_test = K_test * omega;
        p_test = 1 ./ (1 + exp(-a_test));
    
        % Convert probabilities to class labels (threshold at 0.5)
        predictions = ones(size(p_test));
        predictions(p_test < 0.5) = -1;
    
        % Compute accuracy
        accuracy = mean(predictions == TestY);
    end