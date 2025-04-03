clear; clc; close all

%% INPUT:
filename = 'ColorEmotionData.xlsx';
preference_keyword = 'like-dislike';

%% Step1. Color Emotion Mapping
XYZ_db = readtable(filename, ...
    'VariableNamingRule', 'preserve', 'ReadRowNames', true, 'Sheet', 'XYZ');
XYZ_db = [XYZ_db.X XYZ_db.Y XYZ_db.Z];
XYZ = XYZ_db(1:end-1,:);
XYZn = XYZ_db(end,:);

% For color visualization in figure
RGB = max(min(xyz2rgb(XYZ./XYZn(2), "WhitePoint",XYZn./XYZn(2)),1),0);

Emotions = readtable(filename, ...
    'VariableNamingRule', 'preserve', 'ReadRowNames', true, 'Sheet', 'Data');
var_emotion = Emotions.Properties.VariableNames;

% Emotion assessment survey data should be z-score normalized before PCA
% This input data should be normalized z-score value.
X = Emotions.Variables;  

% coeff = eigenvector | score = pc value (raw data*coeff) | latent = eigenvale
[coeff, score, latent] = pca(X);  % Perform PCA on the z-scored data

percent = latent ./ sum(latent); 
cumpercent = cumsum(percent);

figure('WindowState','maximized'); 

% Ratio of explained data with number of principal component
subplot(2,2,1); pareto(percent); 
xlabel('Principal Component'); ylabel('Ratio'); title('Pareto chart')

% Map on 2-D and 3-D space according to the number of components
subplot(2,2,2);
h = biplot(coeff(:, 1:2), 'Scores', score(:, 1:2), 'VarLabels', var_emotion);
xlabel("PC 1"); ylabel("PC 2"); title("Color-Emotion 2D Map")
num_patches = size(X, 1);
num_emotions = size(X, 2);
for i = num_emotions+1:num_emotions*2
    h(i).Marker = 'none';
end
color = RGB; 
for k = 1:num_patches
    h(k + num_emotions*3).MarkerEdgeColor = color(k, :);
    h(k + num_emotions*3).MarkerSize = 16;
end
numPC = 2;

if cumpercent(2) < 0.8
    subplot(2,2,3);
    h = biplot(coeff(:, 1:3), 'Scores', score(:, 1:3), 'VarLabels', var_emotion);
    xlabel("PC 1"); ylabel("PC 2"); zlabel("PC 3");
    title('Color-Emotion 3D Map');
    num_patches = size(X, 1);
    num_emotions = size(X, 2);
    for i = num_emotions+1:num_emotions*2
        h(i).Marker = 'none';
    end
    color = RGB; 
    for k = 1:num_patches
        h(k + num_emotions*3).MarkerEdgeColor = color(k, :);
        h(k + num_emotions*3).MarkerSize = 16;
    end
    numPC = 3;
end
PC = coeff(:, 1:numPC);

% Correlation heatmap of the original variables
subplot(2,2,4); corrMatrix = corr(X); % Correlation matrix of the original variables
heatmap(var_emotion, var_emotion, corrMatrix, 'ColorbarVisible', 'on');
title('Correlation Matrix of Variables');

Name = arrayfun(@(x) sprintf('PC%d', x), 1:numPC, 'UniformOutput', false);
PC_emotions = array2table(score(:, 1:numPC), 'VariableNames', Name);


%% ---  For clear sepreation  --- %%


%% Step2. Color-Emotion Modeling
if ~exist('clr_emotion', 'var')
    clr_emotion = var_emotion;
end

LCh = computeColorVariables(XYZ, XYZn);  % CIE 1973 LCh

Emotions = readtable(filename, ...
    'VariableNamingRule', 'preserve', 'ReadRowNames', true, 'Sheet', 'Data');
TargetEmo = Emotions.Properties.VariableNames;

matches_ = arrayfun(@(x) find(strcmpi(TargetEmo{x}, clr_emotion), 1), 1:numel(TargetEmo), 'UniformOutput', false);
matches_idx = find(~cellfun(@isempty, matches_));
matches_equ = cell2mat(matches_(matches_idx));

Emotion = Emotions(:, matches_idx);
Emotion = Emotion(:, matches_equ);

row_names = arrayfun(@(x) sprintf('Sample %d', x), 1:size(Emotion, 1), 'UniformOutput', false);
empirical_emotion = table2array(Emotion);
T_empirical = array2table(empirical_emotion, 'VariableNames', clr_emotion, 'RowNames', row_names);
% disp(T_empirical);

[AssociationModels, equations] = modelColorEmotion_adv(LCh, Emotion);

Y = table2array(Emotion);  % Convert emotion data table to array;

figure('WindowState','maximized');
for i = 1:length(clr_emotion)
    model = AssociationModels{i};
    coeff = model.flightness_coefficients;
    [a,b] = figure_subplotting(length(clr_emotion));
    subplot(a, b, i);  
    lightness_graph_visualize(coeff,clr_emotion{i}); hold on;
    scatter(LCh(:,1), Y(:,i),30,RGB,'filled'); 
end

figure('WindowState','maximized');
for i = 1:length(clr_emotion)
    model = AssociationModels{i};
    coeff = model.fchroma_coefficients;
    [a,b] = figure_subplotting(length(clr_emotion));
    subplot(a, b, i);  
    chroma_graph_visualize(coeff,90,clr_emotion{i}); hold on;
    scatter(LCh(:,2), Y(:,i),30,RGB,'filled'); 
end

figure('WindowState','maximized');
for i = 1:length(clr_emotion)
    model = AssociationModels{i};

    coeff = model.mdl_hue.Coefficients.Estimate;

    [a,b] = figure_subplotting(length(clr_emotion));
    subplot(a, b, i);  
    hue_graph_visualize(coeff,clr_emotion{i}); hold on;
    scatter(LCh(:,3), Y(:,i),30,RGB,'filled'); 
end

TargetEmo = Emotion.Properties.VariableNames;
for i = 1:length(TargetEmo)
    fprintf('%s\n   %s\n', TargetEmo{i}, equations{i});
end

%% ---  For clear sepreation  --- %%
% clear;

%% Step3. Prediction
AssociationModels = modelColorEmotion_adv(LCh, Emotion);
Y = table2array(Emotion);  % Convert emotion data table to array;

predicted_zscore = zeros(size(LCh, 1), length(clr_emotion));
for i = 1:length(clr_emotion)
    model = AssociationModels{i};    
    weights = model.weights;        % coefficient of determination of each model (below 1 means it better predicted by their mu)

    % preprocess input data
    L_prime = LCh(:, 1);
    C_prime = LCh(:, 2);
    a_prime = cosd(LCh(:,3));        % Red-green axis approximation
    b_prime = sind(LCh(:,3));        % Yellow-blue axis approximation
    a_pprime = cosd(2 .* LCh(:,3));   % Higher order hue term
    b_pprime = sind(2 .* LCh(:,3));   % Higher order hue term

    if any(weights(1:3) > 0.5, 1)
        model = AssociationModels{i};

        att = find(weights(1:3) > 0.5, 1);
        if att == 1
            fprintf(['Lightness Model:', clr_emotion{i},'\n']);
            y_pred = polyval(model.mdl_lightness, L_prime);
        elseif att == 2
            fprintf(['Chroma Model:', clr_emotion{i},'\n']);
            y_pred = polyval(model.mdl_chroma, C_prime);
        elseif att == 3
            fprintf(['Hue Model:', clr_emotion{i},'\n']);
            X = [a_prime, b_prime, a_pprime, b_pprime];
            y_pred = predict(model.mdl_hue, X);
        end

    else
        domi_hue = model.hue_offset; 
        h_prime = cosd(LCh(:, 3) - domi_hue(1));        % cos(h - domi_hue1)
        h_pprime = cosd(2 * (LCh(:, 3) - domi_hue(2))); % cos(2(h - domi_hue2))

        X_pred = [ones(size(LCh, 1), 1), L_prime, C_prime, h_prime, h_pprime];
        y_pred = X_pred * model.fall_coefficients;
    end

    predicted_zscore(:, i) = y_pred;
end

% Evaluating model prediction performance
r_squared_values = zeros(1, length(clr_emotion));
empirical_zscore = Y;
for i = 1:length(clr_emotion)
    predicted = predicted_zscore(:, i);
    actual = empirical_zscore(:, i);
    ss_tot = sum((actual - mean(actual)).^2); 
    ss_res = sum((actual - predicted).^2);  
    r_squared_values(i) = 1 - (ss_res / (ss_tot+eps));
end

disp('R² for each emotion:');
disp(r_squared_values);

emotion_idx = find(~strcmpi(clr_emotion, 'Like-Dislike'));
preference_idx = find(strcmpi(clr_emotion, 'like-dislike'));

figure('WindowState','maximized');
for i = emotion_idx
    [a,b] = figure_subplotting(length(emotion_idx));
    subplot(a, b, i);  
    scatter(predicted_zscore(:,i), empirical_zscore(:,i), 30, RGB, 'filled'); hold on;
    plot([-2.5, 2.5], [-2.5, 2.5], 'k-'); hold off; grid on;
    title(upper(clr_emotion{i})); 
    xlabel('Predicted Zscore'); ylim([-2.5 2.5]);
    ylabel('Empirical Zscore'); xlim([-2.5 2.5]);
    text(-2,2,sprintf('R² = %.4f',r_squared_values(i)))
end
figure; scatter(predicted_zscore(:,preference_idx), empirical_zscore(:,preference_idx), 30, RGB, 'filled'); hold on;
plot([-2.5, 2.5], [-2.5, 2.5], 'k-'); hold off; grid on;
title(upper(clr_emotion{preference_idx})); 
xlabel('Predicted Zscore'); ylim([-2.5 2.5]);
ylabel('Empirical Zscore'); xlim([-2.5 2.5]);
text(-2,2,sprintf('R² = %.4f',r_squared_values(preference_idx)))

%% ---  For clear sepreation  --- %%
% clear;


%% Trunk
% [AssociationModels, equations] = modelColorEmotion_MRM_2(LCh, Emotion);
% TargetEmo = Emotion.Properties.VariableNames;
% for i = 1:length(TargetEmo)
%     fprintf('%s\n   %s\n', TargetEmo{i}, equations{i});
% end
% 
% figure('WindowState','maximized');
% for i = 1:length(clr_emotion)
%     model = AssociationModels{i};
% 
%     coeff = model.fhue_coefficients;
%     hue_offset = model.fhue_offset;
% 
%     [a,b] = figure_subplotting(length(clr_emotion));
%     subplot(a, b, i);  
%     hue_graph_visualize_2(coeff,hue_offset,clr_emotion{i}); hold on;
%     scatter(LCh(:,3), Y(:,i),30,RGB,'filled'); 
% end
% 
% %% Model code
% predicted_zscore = zeros(size(LCh, 1), length(clr_emotion));
% for i = 1:length(clr_emotion)
%     model = AssociationModels{i};    
% 
%     domi_hue = model.fhue_offset;
%     weights = model.weights;
% 
%     % preprocess input data
%     L_prime = LCh(:, 1)./100;
%     C_prime = LCh(:, 2)./100;
%     h_prime = cosd(LCh(:, 3) - domi_hue(1));        % cos(h - domi_hue1)
%     h_2prime = cosd(2 * (LCh(:, 3) - domi_hue(2))); % cos(2(h - domi_hue2))
% 
%     % Create prediction input (L*, C*, cos(h), cos(2h))
%     X_lightness = [L_prime.^2, L_prime ones(size(L_prime))];
%     X_chroma = [C_prime.^2, C_prime ones(size(C_prime))];
%     X_hue = [h_prime, h_2prime];
% 
%     y_fLightness = sum(model.flightness_coefficients'.* X_lightness, 2);
%     y_fChroma = sum(model.fchroma_coefficients' .* X_chroma, 2);
%     y_fHue = sum(model.fhue_coefficients'.* X_hue, 2);
% 
%     y_pred = weights(1)*y_fLightness + weights(2)*y_fChroma + weights(3)*y_fHue;
% 
%     predicted_zscore(:, i) = y_pred;
% end
% 
% % Evaluating model prediction performance
% r_squared_values = zeros(1, length(clr_emotion));
% empirical_zscore = Y;
% for i = 1:length(clr_emotion)
%     predicted = predicted_zscore(:, i);
%     actual = empirical_zscore(:, i);
%     ss_tot = sum((actual - mean(actual)).^2); 
%     ss_res = sum((actual - predicted).^2);  
%     r_squared_values(i) = 1 - (ss_res / (ss_tot+eps));
% end
% 
% disp('R² for each emotion:');
% disp(r_squared_values);
% 
% PredictModels = modelColorEmotion_Pred(LCh, Emotion);
% Y = table2array(Emotion);  % Convert emotion data table to array;
% 
% predicted_zscore = zeros(size(LCh, 1), length(clr_emotion));
% for i = 1:length(clr_emotion)
%     model = PredictModels{i};    % Model coefficients
%     domi_hue = model.hue_offset;   % Dominant hue values for the model
% 
%     % preprocess input data
%     L_prime = LCh(:, 1);
%     C_prime = LCh(:, 2);
%     h_prime = cosd(LCh(:, 3) - domi_hue(1));        % cos(h - domi_hue1)
%     h_pprime = cosd(2 * (LCh(:, 3) - domi_hue(2))); % cos(2(h - domi_hue2))
% 
%     X_pred = [ones(size(LCh, 1), 1), L_prime, C_prime, h_prime, h_pprime];
%     y_pred = X_pred * model.coefficients;
% end
% 
% % Evaluating model prediction performance
% r_squared_values = zeros(1, length(clr_emotion));
% empirical_zscore = Y;
% for i = 1:length(clr_emotion)
%     predicted = predicted_zscore(:, i);
%     actual = empirical_zscore(:, i);
%     ss_tot = sum((actual - mean(actual)).^2); 
%     ss_res = sum((actual - predicted).^2);  
%     r_squared_values(i) = 1 - (ss_res / (ss_tot+eps));
% end
% % Evaluating model prediction performance
% r_squared_values = zeros(1, length(clr_emotion));
% empirical_zscore = Y;
% for i = 1:length(clr_emotion)
%     predicted = predicted_zscore(:, i);
%     actual = empirical_zscore(:, i);
%     ss_tot = sum((actual - mean(actual)).^2); 
%     ss_res = sum((actual - predicted).^2);  
%     r_squared_values(i) = 1 - (ss_res / (ss_tot+eps));
% end
% 
% disp('R² for each emotion:');
% disp(r_squared_values);
% %% Predict new value
% 
% % LCh_test = [LCh(5,1)+20, LCh(2,2)-10, LCh(4,3)];
% LCh_test = LCh;
% 
% prediction = zeros(size(LCh_test, 1), length(clr_emotion));
% for i = 1:length(clr_emotion)
%     model = PredictModels{i}.coefficients;    % Model coefficients
%     domi_hue = PredictModels{i}.hue_offset;   % Dominant hue values for the model
% 
%     % input data
%     L_prime = (LCh_test(:, 1) - mean(LCh(:,1))) ./ std(LCh(:,1));
%     C_prime = (LCh_test(:, 2) - mean(LCh(:,2))) ./ std(LCh(:,2));
%     h_prime = cosd(LCh_test(:, 3) - domi_hue(1));   % cos(h - domi_hue1)
%     h_2prime = cosd(2 * (LCh_test(:, 3) - domi_hue(2))); % cos(2(h - domi_hue2))
% 
%     % Create prediction input (L*, C*, cos(h), cos(2h))
%     X_pred = [ones(size(LCh_test, 1), 1), L_prime, C_prime, h_prime, h_2prime];
%     y_pred = X_pred * model;
% 
%     prediction(:, i) = y_pred;
% end


%% Subfunctions
function [split1, split2] = figure_subplotting(n)
    split1 = floor(sqrt(n));
    split2 = ceil(n / split1);
end