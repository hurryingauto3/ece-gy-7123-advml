function plot_iteration_convergence(run_histories, run_params, plot_title, x_label, y_label, y_data_field, use_log_scale)
    % Plots convergence history (cost or grad_norm) for multiple runs and saves the figure.
    % run_histories: cell array of structures containing history data
    % run_params: cell array of strings for parameter labels
    % plot_title: title of the plot
    % x_label: label for x-axis
    % y_label: label for y-axis
    % y_data_field: field name in history structure to plot
    % use_log_scale: boolean to use log scale for y-axis

    if isempty(run_histories) || isempty(run_params) || length(run_histories) ~= length(run_params)
        fprintf('Warning: History or parameter data missing or mismatched for %s. Skipping plot.\n', plot_title);
        return;
    end

    fig = figure('Name', plot_title, 'Visible', 'off');
    hold on;
    max_len = 0;
    plotted_something = false; % Flag to check if any data was actually plotted

    for i = 1:length(run_histories)
        history = run_histories{i};
        if isstruct(history) && isfield(history, y_data_field) && ~isempty(history.(y_data_field))
            data_vector = history.(y_data_field);
            data_vector(isinf(data_vector)) = NaN; % Handle Inf
            data_vector(imag(data_vector) ~= 0) = NaN; % Handle complex

            if all(isnan(data_vector)), continue; end % Skip if all NaN

            x_values = 0:(length(data_vector)-1);
            max_len = max(max_len, length(x_values));
            plot(x_values, data_vector, 'LineWidth', 1);
            plotted_something = true;
        else
                fprintf('Warning: Invalid history data for run %d in %s.\n', i, plot_title);
        end
    end

    hold off;

    if ~plotted_something
        fprintf('Warning: No valid data plotted for "%s". Skipping save.\n', plot_title);
        close(fig);
        return;
    end

    title(plot_title);
    xlabel(x_label);
    ylabel(y_label);
    if max_len > 0
        xlim([0 max_len-1]);
    end
    grid on;

    if use_log_scale
        set(gca, 'YScale', 'log');
    end

    % Add legend - limit size
    if length(run_params) <= 15
        legend(run_params, 'Location', 'best', 'Interpreter', 'none');
    else
        legend('show');
        disp(['Info: Legend labels suppressed for ', plot_title, ' due to large number of runs (', num2str(length(run_params)), ')']);
    end

    % --- Saving ---
    filename_base = lower(strrep(plot_title, ' ', '_'));
    filename_base = regexprep(filename_base, '[^a-z0-9_]', '');
    filename_png = [filename_base, '.png'];
    try
        saveas(fig, filename_png);
        fprintf('Saved plot: %s\n', filename_png);
    catch ME
        fprintf('Error saving plot %s: %s\n', filename_png, ME.message);
    end
    close(fig);

end