function [normalized_points, T] = normalize_points(points)
    % Calculate centroid of the points
    centroid = mean(points);
    
    % Translate points to have the centroid at the origin
    translated_points = points - centroid;
    
    % Calculate the average distance from the origin
    avg_dist = mean(sqrt(sum(translated_points.^2, 2)));
    
    % Scale factor to make average distance sqrt(2)
    scale = sqrt(2) / avg_dist;
    
    % Construct normalization transformation matrix
    T = [scale, 0, -scale * centroid(1);
         0, scale, -scale * centroid(2);
         0, 0, 1];
    
    % Convert points to homogeneous coordinates
    points_homogeneous = [points, ones(size(points, 1), 1)];
    
    % Apply the normalization transformation
    normalized_points_homogeneous = (T * points_homogeneous')';
    
    % Return normalized points in Cartesian coordinates
    normalized_points = normalized_points_homogeneous(:, 1:2);
end

% Normalize world points and image points
