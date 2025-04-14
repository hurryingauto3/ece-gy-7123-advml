function plot_convergence(results, title_suffix)
    % Plots cost and gradient norm vs time for different optimizers and saves the plot.
    % Inputs:
    %   results     - Struct containing results from different optimizers
    %   title_suffix - Suffix for the plot title and filename
    % Outputs:
    %   Saves a PNG file with the plot

    fig = figure('Name', ['Convergence Comparison' title_suffix], 'Visible', 'off'); % Create figure but keep it hidden initially

    % Plot Cost vs Time
    subplot(2, 1, 1);
    hold on;
    optimizer_names = fieldnames(results);
    valid_optimizers = {}; % Store names of optimizers actually plotted
    for i = 1:length(optimizer_names)
        name = optimizer_names{i};
        if isfield(results.(name), 'history') && ~isempty(results.(name).history) && isfield(results.(name).history, 'time')
            plot(results.(name).history.time, results.(name).history.cost, 'LineWidth', 1.5);
            valid_optimizers{end+1} = name; % Add to list for legend
        end
    end
    hold off;
    xlabel('Time (seconds)');
    ylabel('Cost J(\omega)');
    title(['Cost Function vs. Time' title_suffix]);
    if ~isempty(valid_optimizers), legend(valid_optimizers, 'Location', 'best'); end
    grid on;
    set(gca, 'YScale', 'log');

    % Plot Gradient Norm vs Time
    subplot(2, 1, 2);
    hold on;
    valid_optimizers = {};
    for i = 1:length(optimizer_names)
        name = optimizer_names{i};
            if isfield(results.(name), 'history') && ~isempty(results.(name).history) && isfield(results.(name).history, 'time')
            plot(results.(name).history.time, results.(name).history.grad_norm, 'LineWidth', 1.5);
            valid_optimizers{end+1} = name;
            end
    end
    hold off;
    xlabel('Time (seconds)');
    ylabel('||Grad J(\omega)||');
    title(['Gradient Norm vs. Time' title_suffix]);
    if ~isempty(valid_optimizers), legend(valid_optimizers, 'Location', 'best'); end
    grid on;
    set(gca, 'YScale', 'log');

    % --- Saving ---
    filename_base = ['convergence_comparison' strrep(strrep(title_suffix,' ','_'),'(','')]; % Basic filename
    filename_png = [filename_base, '.png'];
    try
        saveas(fig, filename_png);
        fprintf('Saved plot: %s\n', filename_png);
    catch ME
        fprintf('Error saving plot %s: %s\n', filename_png, ME.message);
    end
    close(fig); % Close the figure after saving (optional, keeps screen clean)

end