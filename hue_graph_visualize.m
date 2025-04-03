function hue_graph_visualize(coeff, emotion)
% Define the range of hue (h) values
h = 0:1:360;  % Hue in degrees

% Define the equation coefficients
% coeff = model.mdl_hue.Coefficients.Estimate

% Evaluate the equation
y = coeff(1) + ...
    coeff(2) * cosd(h) + coeff(3) * sind(h) + ...
    coeff(4) * cosd(2*h) + coeff(5) * sind(2*h);

% Generate a color map based on hue
lab_colors = [repmat(70, length(h), 1), (127.*cosd(h))', (127.*sind(h))']; % 70 for gray, 127 for max saturation
colors = max(min(lab2rgb(lab_colors), 1), 0); % Convert Lab to RGB using colorspace conversion

% Plot the graph with color mapping
hold on;
for i = 1:length(h) - 1
    % Plot each segment with its corresponding color
    plot(h(i:i+1), y(i:i+1), 'Color', colors(i, :), 'LineWidth', 2);
end

% Customize the plot
xlabel('Hue (h°)', 'FontSize', 12);

text = sprintf('%.4f + %.4f * cos(h) + %.4f * sin(h)+ %.4f * cos(2h) + %.4f * sin(2h)', ...
        coeff(1), coeff(2),coeff(3), coeff(4), coeff(5));
title(text, 'FontSize', 10, 'FontWeight', 'normal')

xlim([0, 360]);  % Limit x-axis to the range of hue
xticks(0:45:360); % Add ticks every 45 degrees
if length(strsplit(emotion, '-')) > 1
    split_str = strsplit(emotion, '-');
    emotion = ['- ', upper(split_str{2}), ' vs. ', upper(split_str{1}) ,' +'];
end
ylabel(emotion)
ylim([-2.5 2.5])

grid on;
yline(0);
hold off;

end