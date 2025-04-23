function chroma_graph_visualize(coeff, hue, emotion)
% Define the range of chroma (C) values
C = ((0:1:100))'; 

X_chroma = [C.^2, C ones(size(C))];

% Evaluate the equation
y = sum(coeff'.* X_chroma, 2);

% Generate a color map based on hue
lab_colors = [70.*ones(length(C), 1), C.*cosd(hue), C.*sind(hue)]; 
colors = max(min(lab2rgb(lab_colors), 1), 0); % Convert Lab to RGB using colorspace conversion

% Plot the graph with color mapping
hold on;
for i = 1:length(C) - 1
    % Plot each segment with its corresponding color
    plot(C(i:i+1), y(i:i+1), 'Color', colors(i, :), 'LineWidth', 2);
end

% Customize the plot
xlabel('Chroma (C*)', 'FontSize', 12);

text = sprintf('%.2f + %.2f * C* + %.2f * C*^2', ...
                coeff(3), coeff(2), coeff(1));  
% title(text, 'FontSize', 10, 'FontWeight', 'normal')

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