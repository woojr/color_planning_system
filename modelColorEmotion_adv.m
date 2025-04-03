function [models, equations] = modelColorEmotion_adv(patches, emotions)
    
    %% Color appearance attribute data preprocessing (Independent Variables)
    L = patches(:, 1);
    C = patches(:, 2);
    h = patches(:, 3);

    % Sine and cosine transformations for h
    a_prime = cosd(h);        % Red-green axis approximation
    b_prime = sind(h);        % Yellow-blue axis approximation
    a_pprime = cosd(2 * h);   % Higher order hue term
    b_pprime = sind(2 * h);   % Higher order hue term

    % Combine normalized independent variables into a matrix X
    X_lightness = L;
    X_chroma = C;
    X_hue = [a_prime, b_prime, a_pprime, b_pprime];
    X = [L, C, a_prime, b_prime, a_pprime, b_pprime];

    % Prepare dependent variables (emotions)
    data = table2array(emotions);  % Convert emotion data table to array;
    mu = mean(data, 1); 
    sigma = std(data, 0, 1);
    Y = (data - mu) ./ sigma;  % Z-score normalization

    %% Modeling
    models = cell(size(Y, 2), 1);  % To store regression models
    equations = cell(size(Y, 2), 1);  % To store regression equations

    for i = 1:size(Y, 2)
        y = Y(:, i);  % Current emotion variable

        models{i}.mdl_lightness = polyfit(X_lightness, y, 2);
        models{i}.mdl_chroma = polyfit(X_chroma, y, 2);
        models{i}.mdl_hue = fitlm(X_hue, y);   

        %-- Lightness model
        coeffs_mdl_L = models{i}.mdl_lightness;
        %-- Chroma model
        coeffs_mdl_C = models{i}.mdl_chroma;
        %-- Hue model
        coeffs_mdl_h = models{i}.mdl_hue.Coefficients.Estimate;

        %-- Weight
        SS_tot = sum((y - mean(y)).^2);
        pred_L = polyval(coeffs_mdl_L, X_lightness);
        r2_lightness = 1 - (sum((y - pred_L).^2) / SS_tot);
        pred_C = polyval(coeffs_mdl_C, X_chroma);
        r2_chroma = 1 - (sum((y - pred_C).^2) / SS_tot);   
        r2_hue = models{i}.mdl_hue.Rsquared.Ordinary;       

        models{i}.normLightness_ma = [mean(L); std(L)];
        models{i}.normChroma_ma = [mean(C); std(C)];

        models{i}.flightness_coefficients = coeffs_mdl_L';
        models{i}.fchroma_coefficients = coeffs_mdl_C';
        models{i}.fhue_coefficients = coeffs_mdl_h;

%% 
        models{i}.mdl_all = fitlm(X, y);
        coeffs = models{i}.mdl_all.Coefficients.Estimate;

        % Compute transformed coefficients for interpretation
        coeffs_mdl_all = [coeffs(1); ...  % Intercept
            coeffs(2:3); ...                  % L* and C*
            hypot(coeffs(4), coeffs(5)); ... % Magnitude for cos(h), sin(h)
            hypot(coeffs(6), coeffs(7))];    % Magnitude for cos(2h), sin(2h)

        % Calculate hue angles from cosine and sine terms
        hue_offset = [atan2d(coeffs(5), coeffs(4)), ... % For 1st order hue
                      atan2d(coeffs(7), coeffs(6))];    % For 2nd order hue
        hue_offset = round(mod(hue_offset, 360));  % Ensure angles are in [0, 360)

        % Save results in the model structure
        models{i}.hue_offset = hue_offset;
        models{i}.fall_coefficients = coeffs_mdl_all;

        % preprocess input data
        h_prime = cosd(h - hue_offset(1));        % cos(h - dominant angle)
        h_pprime = cosd(2 * (h - hue_offset(2))); % cos(2(h - dominant angle))
    
        % Create prediction input (L*, C*, cos(h), cos(2h))
        pred_X = [ones(size(patches, 1), 1), L, C, h_prime, h_pprime];
        pred_Y = pred_X * models{i}.fall_coefficients;
        r2_all = 1 - (sum((y - pred_Y).^2) / SS_tot);   

        weights = [r2_lightness; r2_chroma; r2_hue; r2_all];
        models{i}.weights = weights;

        L_term = sprintf('Lightness: %.4f, f{ %.4f + %.4f * L + %.4f * L^2 }', ...
                        weights(1), coeffs_mdl_L(3), coeffs_mdl_L(2), coeffs_mdl_L(1));  
        C_term = sprintf('Chroma:    %.4f, f{ %.4f + %.4f * C + %.4f * C^2 }', ...
                        weights(2), coeffs_mdl_C(3), coeffs_mdl_C(2), coeffs_mdl_C(1));  
        hue_term = sprintf('Hue:       %.4f, f{ %.4f + %.4f * cos(h) + %.4f * sin(h)+ %.4f * cos(2h) + %.4f * sin(2h) }', ...
                        weights(3), coeffs_mdl_h(1), coeffs_mdl_h(2),coeffs_mdl_h(3), coeffs_mdl_h(4), coeffs_mdl_h(5));
        all_term = sprintf('Color:     %.4f, f{ %.4f + %.4f * L + %.4f * C + %.4f * cos(h - %d) + %.4f * cos(2(h - %d)) }', ...
                        weights(4), coeffs_mdl_all(1), coeffs_mdl_all(2), coeffs_mdl_all(3), ...
                        coeffs_mdl_all(4), hue_offset(1), coeffs_mdl_all(5), hue_offset(2));

        equations{i} = sprintf(['\t %s  \n' ...
                                '\t %s  \n' ...
                                '\t %s  \n' ...
                                '\t %s'], ... 
                                L_term, C_term, hue_term, all_term);
    end

    % % Print the regression equations
    % for i = 1:length(TargetEmo)
    %     fprintf('%s\n   %s\n', TargetEmo{i}, equations{i});
    % end

end
