function plot_accuracy_hyperparam(hyperparams1, hyperparams2, acc_matrix, label1, label2, title_str)
    % Plots accuracy vs. hyperparameter 1 for different values of hyperparameter 2 and saves the plot.
    % Inputs:
    %   hyperparams1 - Vector of hyperparameter 1 values
    %   hyperparams2 - Vector of hyperparameter 2 values
    %   acc_matrix   - Matrix of accuracies (rows: hyperparams1, columns: hyperparams2)
    %   label1       - Label for hyperparameter 1
    %   label2       - Label for hyperparameter 2
    %   title_str    - Title for the plot
    % Outputs:
    %   Saves a PNG file with the plot
    
        fig = figure('Name', title_str, 'Visible', 'off');
        hold on;
        markers = {'-o', '-s', '-^', '-d', '-v', '-x', '-+', '-*'}; % More markers
        num_groups = length(hyperparams2);
        if isempty(hyperparams1) || isempty(hyperparams2) || isempty(acc_matrix)
            fprintf('Warning: Empty data for plot "%s". Skipping.\n', title_str);
            close(fig);
            return;
        end
    
        for j = 1:num_groups
            plot(hyperparams1, acc_matrix(:,j) * 100, markers{mod(j-1, length(markers))+1}, 'LineWidth', 1.5, 'MarkerSize', 6);
        end
        hold off;
        xlabel(label1);
        ylabel('Test Accuracy (%)');
        title(title_str);
        legend_labels = arrayfun(@(p) sprintf('%s = %g', label2, p), hyperparams2, 'UniformOutput', false);
        legend(legend_labels, 'Location', 'best');
        grid on;
        % Make x-axis log scale only if values span more than factor of 100
        if max(hyperparams1) / min(hyperparams1) > 100
             set(gca, 'XScale', 'log');
        end
    
        % --- Saving ---
        % Create filename from title, replacing spaces and invalid chars
        filename_base = lower(strrep(title_str, ' ', '_'));
        filename_base = regexprep(filename_base, '[^a-z0-9_]', ''); % Keep only letters, numbers, underscore
        filename_png = [filename_base, '.png'];
         try
            saveas(fig, filename_png);
            fprintf('Saved plot: %s\n', filename_png);
        catch ME
            fprintf('Error saving plot %s: %s\n', filename_png, ME.message);
        end
        close(fig);
    
    end