function [models, equations] = modelColorEmotion_LCh(patches, emotions)

    %% Color appearance attribute data preprocessing (Independent Variables)
    L = patches(:,1);
    C = patches(:,2);
    h = patches(:,3);

    % Sine and cosine transformations for h
    a_prime = cosd(h);        % cos(h)
    b_prime = sind(h);        % sin(h)
    a_pprime = cosd(2 * h);   % cos(2h)
    b_pprime = sind(2 * h);   % sin(2h)

    % Combine independent variables into a matrix X (no normalization)
    X_lightness = L;
    X_chroma = C;
    X_hue = [a_prime, b_prime, a_pprime, b_pprime];
    X_color = [L, C, a_prime, b_prime, a_pprime, b_pprime];

    assert(all(~isnan(X_color(:))) && all(~isinf(X_color(:))), 'X contains NaN or Inf');

    Y = table2array(emotions);

    %% Modeling
    models = cell(size(Y, 2), 1);
    equations = cell(size(Y, 2), 1);

    for i = 1:size(Y, 2)
        y = Y(:, i);

        models{i}.mdl_lightness = polyfit(X_lightness, y, 2);
        models{i}.mdl_chroma = polyfit(X_chroma, y, 2);
        models{i}.mdl_hue = fitlm(X_hue, y);

        coeffs_mdl_J = models{i}.mdl_lightness;
        coeffs_mdl_C = models{i}.mdl_chroma;
        coeffs_mdl_h = models{i}.mdl_hue.Coefficients.Estimate;

        % R² for individual models
        SS_tot = sum((y - mean(y)).^2);
        r2_lightness = 1 - sum((y - polyval(coeffs_mdl_J, X_lightness)).^2) / SS_tot;
        r2_chroma = 1 - sum((y - polyval(coeffs_mdl_C, X_chroma)).^2) / SS_tot;
        r2_hue = models{i}.mdl_hue.Rsquared.Ordinary;

        models{i}.normLightness_ma = [mean(L); std(L)];
        models{i}.normChroma_ma = [mean(C); std(C)];
        models{i}.flightness_coefficients = coeffs_mdl_J';
        models{i}.fchroma_coefficients = coeffs_mdl_C';
        models{i}.fhue_coefficients = coeffs_mdl_h;

        %% Full JCh model
        X_color = [L, C, a_prime, b_prime, a_pprime, b_pprime];
        models{i}.mdl_all = fitlm(X_color, y);
        coeffs = models{i}.mdl_all.Coefficients.Estimate;
        
        % 변환된 계수: 해석용 (예측용 아님)
        coeffs_mdl_all = [coeffs(1); ...
                          coeffs(2:3); ...
                          hypot(coeffs(4), coeffs(5)); ...
                          hypot(coeffs(6), coeffs(7))];
        
        hue_offset = [atan2d(coeffs(5), coeffs(4)), ...
                      atan2d(coeffs(7), coeffs(6))];
        hue_offset = round(mod(hue_offset, 360));
        
        models{i}.hue_offset = hue_offset;
        models{i}.fall_coefficients = coeffs_mdl_all;
        
        y_pred = predict(models{i}.mdl_all, X_color);
        SS_res = sum((y - y_pred).^2);
        r2_all = 1 - SS_res / SS_tot;

        %% Weight aggregation
        weights = [r2_lightness; r2_chroma; r2_hue; r2_all];
        models{i}.weights = weights;

        %% Equation description
        L_term = sprintf('Lightness: %.4f, f{ %.4f + %.4f * J + %.4f * J^2 }', ...
                         weights(1), coeffs_mdl_J(3), coeffs_mdl_J(2), coeffs_mdl_J(1));
        C_term = sprintf('Chroma:    %.4f, f{ %.4f + %.4f * C + %.4f * C^2 }', ...
                         weights(2), coeffs_mdl_C(3), coeffs_mdl_C(2), coeffs_mdl_C(1));
        hue_term = sprintf('Hue:       %.4f, f{ %.4f + %.4f * cos(h) + %.4f * sin(h)+ %.4f * cos(2h) + %.4f * sin(2h) }', ...
                         weights(3), coeffs_mdl_h(1), coeffs_mdl_h(2), coeffs_mdl_h(3), coeffs_mdl_h(4), coeffs_mdl_h(5));
        all_term = sprintf('Color:     %.4f, f{ %.4f + %.4f * J + %.4f * C + %.4f * cos(h - %d) + %.4f * cos(2(h - %d)) }', ...
                         weights(4), coeffs_mdl_all(1), coeffs_mdl_all(2), coeffs_mdl_all(3), ...
                         coeffs_mdl_all(4), hue_offset(1), coeffs_mdl_all(5), hue_offset(2));

        equations{i} = sprintf('\t %s\n\t %s\n\t %s\n\t %s', ...
                                L_term, C_term, hue_term, all_term);
    end
end
