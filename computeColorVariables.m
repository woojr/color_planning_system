function patches = computeColorVariables(XYZ, XYZn)

% Compute color appearance coordinates: CIELAB
Lab = XYZ2Lab(XYZ, XYZn);
LabCh = Lab2LabCh(Lab);

patches = LabCh(:,[1,4,5]);

end


%% Subfunctions
function Lab = XYZ2Lab(XYZ, XYZn)
    % Convert from XYZ to LAB, using whitepoint XYZn.
    % Input can be Nx3 or 3xN.
    
    % check size
    if ~any(size(XYZ)==3)
        error('XYZ must be either Nx3 or 3xN');
    end
    % check orientation
    switched = size(XYZ,2) ~= 3;
    if switched
        XYZ = XYZ'; % make Nx3
    end
    if size(XYZ,2) == size(XYZ,1)
        warning('Square matrix assumed to be Nx3 orientation.')
    end
    
    Lab = zeros(size(XYZ));
    
    XYZ = XYZ ./ XYZn(:)';
    Lab(:,1) = 116 * f(XYZ(:, 2)) - 16;
    Lab(:,2) = 500 * (f(XYZ(:, 1)) - f(XYZ(:, 2)));
    Lab(:,3) = 200 * (f(XYZ(:, 2)) - f(XYZ(:, 3)));
    
    if switched
        Lab = Lab';
    end
end

function y = f(x)
    % See Bruce Lindbloom http://brucelindbloom.com/index.html?Eqn_Luv_to_XYZ.html
    % for the explanation of this implementation.
    
    mask = x <= 216/24389;
    y = x .^ (1/3);
    y(mask) = (x(mask) * 24389/27 + 16) / 116;
end

function LabCh = Lab2LabCh( Lab )
    LabCh(:,1:3) = Lab;                                       % Lightness, chromaticity coordinate
    LabCh(:,4) = hypot(Lab(:,2),Lab(:,3));                    % Chroma
    LabCh(:,5) = atan2(Lab(:,3),Lab(:,2)).*(180/pi);          % hue angle
    LabCh(:,5) = LabCh(:,5)+(LabCh(:,5)<0).*360;              
end
