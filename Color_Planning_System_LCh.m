clear; clc; close all

%% Step1. Color Emotion Assessment
filename = 'ColorEmotionData.xlsx';
preference_keyword = 'Like-Dislike';
mode = 'related';  % ['all' | 'related']

% This is new color appearance that will be predicted using model
test_XYZ = [347.43	351.43	428.33; ...
            401.17	331.57	206.30; ...
            138.07	164.83	100.77; ...
            149.23	154.90	181.10;]; % XYZ value

%% Step2. Color Emotion Mapping
XYZ_db = readtable(filename, ...
    'VariableNamingRule', 'preserve', 'ReadRowNames', true, 'Sheet', 'XYZ');
XYZ_db = [XYZ_db.X XYZ_db.Y XYZ_db.Z];
XYZ = XYZ_db(1:end-1,:);
XYZn = XYZ_db(end,:);
RGB = max(min(xyz2rgb(XYZ./XYZn(2), "WhitePoint",XYZn./XYZn(2)),1),0);

test_LCh = computeLCh(test_XYZ, XYZn);
test_RGB = max(min(xyz2rgb(test_XYZ./XYZn(2), "WhitePoint", XYZn./XYZn(2)),1),0);

Emotions = readtable(filename, ...
    'VariableNamingRule', 'preserve', 'ReadRowNames', true, 'Sheet', 'Data');
var_emotion = Emotions.Properties.VariableNames;
X = Emotions.Variables;  

[coeff, score, latent] = pca(X);
percent = latent ./ sum(latent); 
cumpercent = cumsum(percent);

figure('WindowState','maximized'); 
subplot(2,2,1); pareto(percent); 
xlabel('Principal Component'); ylabel('Ratio'); title('Pareto chart')

subplot(2,2,2);
h = biplot(coeff(:, 1:2), 'Scores', score(:, 1:2), 'VarLabels', var_emotion);
xlabel("PC 1"); ylabel("PC 2"); title("Color-Emotion 2D Map")
num_patches = size(X, 1);
num_emotions = size(X, 2);
for i = num_emotions+1:num_emotions*2
    h(i).Marker = 'none';
end
for k = 1:num_patches
    h(k + num_emotions*3).MarkerEdgeColor = RGB(k, :);
    h(k + num_emotions*3).MarkerSize = 16;
end
numPC = 2;

if cumpercent(2) < 0.8
    subplot(2,2,3);
    h = biplot(coeff(:, 1:3), 'Scores', score(:, 1:3), 'VarLabels', var_emotion);
    xlabel("PC 1"); ylabel("PC 2"); zlabel("PC 3");
    title('Color-Emotion 3D Map');
    for i = num_emotions+1:num_emotions*2
        h(i).Marker = 'none';
    end
    for k = 1:num_patches
        h(k + num_emotions*3).MarkerEdgeColor = RGB(k, :);
        h(k + num_emotions*3).MarkerSize = 16;
    end
    numPC = 3;
end
PC = coeff(:, 1:numPC);

subplot(2,2,4); corrMatrix = corr(X); 
heatmap(var_emotion, var_emotion, corrMatrix, 'ColorbarVisible', 'on');
title('Correlation Matrix of Variables');

Name = arrayfun(@(x) sprintf('PC%d', x), 1:numPC, 'UniformOutput', false);
PC_emotions = array2table(score(:, 1:numPC), 'VariableNames', Name);

if strcmpi(mode, 'related')
    like_idx = find(strcmpi(var_emotion, preference_keyword));
    if isempty(like_idx)
        warning('Like-Dislike not found.');
        selected_idx = 1:length(var_emotion);
    else
        like_corr = corrMatrix(:, like_idx);
        like_corr(like_idx) = 0;
        [~, top_idx] = maxk(abs(like_corr), 3);
        selected_idx = unique([like_idx; top_idx]);

        fprintf('\nTop 3 emotions most correlated with "%s":\n', preference_keyword);
        for i = 1:3
            idx = top_idx(i);
            fprintf('  %s: r = %.4f\n', var_emotion{idx}, corrMatrix(idx, like_idx));
        end
    end
else
    selected_idx = 1:length(var_emotion);
end

%% Step3. Color Emotion Modeling
clr_emotion = var_emotion(selected_idx);
Emotion = Emotions(:, clr_emotion);
Y = table2array(Emotion);

LCh = computeLCh(XYZ, XYZn);
[AssociationModels, equations] = modelColorEmotion_LCh(LCh, Emotion);

figure('WindowState','maximized');
for i = 1:length(clr_emotion)
    model = AssociationModels{i};
    coeff = model.flightness_coefficients;
    [a,b] = figure_subplotting(length(clr_emotion));
    subplot(a, b, i);  
    lightness_graph_visualize(coeff, clr_emotion{i}); hold on;
    scatter(LCh(:,1), Y(:,i), 70, RGB, 'filled'); 
end

figure('WindowState','maximized');
for i = 1:length(clr_emotion)
    model = AssociationModels{i};
    coeff = model.fchroma_coefficients;
    [a,b] = figure_subplotting(length(clr_emotion));
    subplot(a, b, i);  
    chroma_graph_visualize(coeff, 90, clr_emotion{i}); hold on;
    scatter(LCh(:,2), Y(:,i), 70, RGB, 'filled'); 
end

figure('WindowState','maximized');
for i = 1:length(clr_emotion)
    model = AssociationModels{i};
    coeff = model.mdl_hue.Coefficients.Estimate;
    [a,b] = figure_subplotting(length(clr_emotion));
    subplot(a, b, i);  
    hue_graph_visualize(coeff, clr_emotion{i}); hold on;
    scatter(LCh(:,3), Y(:,i), 70, RGB,'filled'); 
end

for i = 1:length(clr_emotion)
    fprintf('%s\n   %s\n', clr_emotion{i}, equations{i});
end

%% Model performance
AssociationModels = modelColorEmotion_LCh(LCh, Emotion);
predicted_zscore = zeros(size(LCh,1), length(clr_emotion));
for i = 1:length(clr_emotion)
    model = AssociationModels{i};    
    weights = model.weights;
    [~, ind_model] = max(model.weights);

    L_prime = LCh(:,1); C_prime = LCh(:,2);
    a_prime = cosd(LCh(:,3)); b_prime = sind(LCh(:,3));
    a_pprime = cosd(2 .* LCh(:,3)); b_pprime = sind(2 .* LCh(:,3));

    if ind_model == 1
        fprintf(['Lightness Model used for ', clr_emotion{i},'\n']);
        y_pred = polyval(model.mdl_lightness, L_prime);
    elseif ind_model == 2
        fprintf(['Chroma Model used for ', clr_emotion{i},'\n']);
        y_pred = polyval(model.mdl_chroma, C_prime);
    elseif ind_model == 3
        fprintf(['Hue Model used for ', clr_emotion{i},'\n']);
        X = [a_prime, b_prime, a_pprime, b_pprime];
        y_pred = predict(model.mdl_hue, X);
    elseif ind_model == 4
        fprintf(['Color Model used for ', clr_emotion{i},'\n']);
        X = [L_prime, C_prime, a_prime, b_prime, a_pprime, b_pprime];
        y_pred = predict(model.mdl_all, X);
    end

    predicted_zscore(:, i) = y_pred;
end

r_squared_values = zeros(1, length(clr_emotion));
empirical_zscore = Y;
for i = 1:length(clr_emotion)
    predicted = predicted_zscore(:, i);
    actual = empirical_zscore(:, i);
    ss_tot = sum((actual - mean(actual)).^2); 
    ss_res = sum((actual - predicted).^2);  
    r_squared_values(i) = 1 - (ss_res / ss_tot);
end

fprintf('\nR\x00b2 for %s %s %s %s:\n', clr_emotion{:});
disp(r_squared_values);

emotion_idx = find(~strcmpi(clr_emotion, preference_keyword));
preference_idx = find(strcmpi(clr_emotion, preference_keyword));

figure('WindowState','maximized');
for i = 1:length(clr_emotion)
    [a,b] = figure_subplotting(length(clr_emotion));
    subplot(a, b, i);  
    scatter(predicted_zscore(:,i), empirical_zscore(:,i), 70, RGB, 'filled'); hold on;
    plot([-2.5, 2.5], [-2.5, 2.5], 'k-'); hold off; grid on;
    title(upper(clr_emotion{i})); 
    xlabel('Predicted Zscore'); ylim([-2.5 2.5]);
    ylabel('Empirical Zscore'); xlim([-2.5 2.5]);
    text(-2,2,sprintf('R\x00b2 = %.4f',r_squared_values(i)))
end

%% Predicting with new color
new_zscore = zeros(size(test_LCh, 1), length(clr_emotion));
for i = 1:length(clr_emotion)
    model = AssociationModels{i};    
    weights = model.weights;
    L_prime = test_LCh(:,1); C_prime = test_LCh(:,2);
    a_prime = cosd(test_LCh(:,3)); b_prime = sind(test_LCh(:,3));
    a_pprime = cosd(2 .* test_LCh(:,3)); b_pprime = sind(2 .* test_LCh(:,3));

    X = [L_prime, C_prime, a_prime, b_prime, a_pprime, b_pprime];
    y_pred = predict(model.mdl_all, X);

    new_zscore(:, i) = y_pred;
end

new_table = array2table(new_zscore, 'VariableNames', clr_emotion);
fprintf('\nPredicted with new samples:\n');
disp(new_table);


%% Subfunctions
function [split1, split2] = figure_subplotting(n)
    split1 = floor(sqrt(n));
    split2 = ceil(n / split1);
end
