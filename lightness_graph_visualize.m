function lightness_graph_visualize(coeff, emotion)
% Define the range of lightness (L) values
L = ((0:1:100))'; 

X_lightness = [L.^2, L ones(size(L))];

% Evaluate the equation
y = sum(coeff'.* X_lightness, 2);

% Generate a color map based on hue
lab_colors = [L, zeros(length(L), 1), zeros(length(L), 1)]; % achromatic
colors = max(min(lab2rgb(lab_colors), 1), 0); % Convert Lab to RGB using colorspace conversion

% Plot the graph with color mapping
hold on;
for i = 1:length(L) - 1
    % Plot each segment with its corresponding color
    plot(L(i:i+1), y(i:i+1), 'Color', colors(i, :), 'LineWidth', 2);
end

% Customize the plot
xlabel('Lightness (L*)', 'FontSize', 12);

text = sprintf('%.2f + %.2f * L + %.2f * L^2', ...
                coeff(3), coeff(2), coeff(1));  
title(text, 'FontSize', 10, 'FontWeight', 'normal')

xlim([0, 100]);  % Limit x-axis to the range of hue
xticks(0:10:100); % Add ticks every 10 level
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