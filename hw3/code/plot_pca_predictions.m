function plot_pca_predictions(TestX, results, TestY, title_suffix)
    % Generates tiled PCA plots for predictions from different optimizers and saves them.
    % Inputs:
    %   TestX: Matrix of test data features (rows = samples, columns = features)
    %   results: Struct containing predictions and accuracies from different optimizers
    %   TestY: Vector of true labels for the test data
    %   title_suffix: Suffix for the plot title and filename
    % Outputs:
    %   Saves a figure with PCA plots of predictions from different optimizers
    
    [pca_coeff, score, ~, ~, explained, ~] = pca(TestX); % Keep coeff if needed later
    optimizer_names = fieldnames(results);
    num_optimizers = length(optimizer_names);
    if num_optimizers == 0, return; end

    % Determine grid layout
    numCols = ceil(sqrt(num_optimizers + 1)); % +1 for true labels plot
    numRows = ceil((num_optimizers + 1) / numCols);

    fig = figure('Name', ['PCA Predictions Comparison' title_suffix], 'Visible', 'off');
    t = tiledlayout(numRows, numCols, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, ['PCA Comparison of Optimizer Predictions' title_suffix]);

    % Plot predictions for each optimizer
    for i = 1:num_optimizers
        name = optimizer_names{i};
        if isfield(results.(name), 'predictions') && ~isempty(results.(name).predictions) && isfield(results.(name), 'accuracy')
            nexttile;
            gscatter(score(:,1), score(:,2), results.(name).predictions);
            acc_str = sprintf('Acc: %.2f%%', results.(name).accuracy * 100);
            title(sprintf('%s (%s)', name, acc_str));
            xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
            ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
            legend off;
        end
    end

    % Plot true labels
    nexttile;
    gscatter(score(:,1), score(:,2), TestY);
    title('True Labels');
    xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
    ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
    legend off;

    % --- Saving ---
    filename_base = ['pca_predictions_comparison' strrep(strrep(title_suffix,' ','_'),'(','')];
    filename_png = [filename_base, '.png'];
    try
        saveas(fig, filename_png);
        fprintf('Saved plot: %s\n', filename_png);
    catch ME
        fprintf('Error saving plot %s: %s\n', filename_png, ME.message);
    end
    close(fig);

end