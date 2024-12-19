% Load intrinsic matrix from camera parameters
%matrice_k = cameraParams.Intrinsics.IntrinsicMatrix';
%disp('Intrinsic matrix K:');
%disp(matrice_k);

% Load and display the image
image = imread("APC_0016.jpg");
imshow(image);
datacursormode on;  % Enable data cursor mode for point inspection

% Define world points and image points
world_points = [2 1;8 1;4 2;6 3;2 4;4 4;8 4;7 5] ;
image_points=[
    1971, 1159;
    1981, 2140;
    1802, 1539;
    1641,1890;
    1415,1202;
    1452, 1566;
    1506, 2193;
    1330, 2059;
];

numPoints = size(world_points, 1);
if size(image_points, 1) ~= numPoints
    error('The number of world points must match the number of image points.');
end

% Step 1: Standardize Points
%[normalized_world_points, T_world] = normalize_points(world_points);
%[normalized_image_points, T_image] = normalize_points(image_points);

% Step 2: Construct Linear System for DLT using normalized points
A = zeros(2 * numPoints, 9);
for i = 1:numPoints
    X = world_points(i, 1);
    Y = world_points(i, 2);
    x = image_points(i, 1);
    y = image_points(i, 2);
    
    A(2*i-1, :) = [-X, -Y, -1, 0, 0, 0, x*X, x*Y, x];
    A(2*i, :) = [0, 0, 0, -X, -Y, -1, y*X, y*Y, y];
end

% Step 3: Solve the Linear System Using S  D to Find Normalized Homography
[~, ~, V] = svd(A);
H = reshape(V(:, end), [3, 3])';
H;

% Step 4: Denormalize the Homography Matrix
%H = inv(T_image) * H_normalized * T_world;
%H = H / H(3, 3);  % Normalize H so that H(3,3) = 1

%fprintf('Denormalized Homography matrix H:\n');
disp(H);

% Step 5: Transform World Points using the Denormalized Homography
world_points_homogeneous = [world_points, ones(numPoints, 1)]';  % Convert to homogeneous
transformed_points_homogeneous = H * world_points_homogeneous;  % Apply homography
transformed_points = transformed_points_homogeneous ./ transformed_points_homogeneous(3, :);  % Normalize

% Extract x and y coordinates of transformed points
x_transformed = transformed_points(1, :);
y_transformed = transformed_points(2, :);

% Step 6: Visualize Original and Transformed (Denormalized) Points
figure;
imshow('APC_0016.jpg');  % Load image for visualization
hold on;
datacursormode on;
plot(image_points(:, 1), image_points(:, 2), 'go', 'MarkerSize', 10, 'LineWidth', 2);  % Original image points in green
plot(x_transformed, y_transformed, 'rx', 'MarkerSize', 10, 'LineWidth', 2); 
% Transformed points in yellow

% Draw lines connecting original points to transformed points for comparison
for i = 1:numPoints
    plot([image_points(i, 1), x_transformed(i)], [image_points(i, 2), y_transformed(i)], 'b--');  % Dashed lines
end

legend('Original Image Points', 'Transformed Points', 'Lines to Transformed Points');
title('Visual Comparison of Original and Transformed Points');
hold off;
%%
K=[1.682793625028055e+03 0 7.807846303429646e+02; 0 1.682351387650792e+03 1.027805128038535e+03; 0 0 1 ]
RT=inv(K)*H;
R1=RT(:,1);
R2=RT(:,2);
R3=cross(R1,R2);
T=RT(:,3);
R=[R1 R2 R3];
D=det(R);
alpha=nthroot(D,4);
Rup=[R1./alpha R2./alpha R3./(alpha.^2)];

Dup=det(Rup);
Tup=T./alpha;
RTup=cat(2,Rup,Tup);
P=K*RTup;
%%

    
world_cube=[0 0 0 1;0 2 0 1; 2 0 0 1; 2 2 0 1;0 0 2 1;0 2 2 1;2 0 2 1;2 2 2 1 ]
cube_2d_homogeneous = P * world_cube'  % Project the 3D points to 2D
cube_2d = cube_2d_homogeneous ./ cube_2d_homogeneous(3, :) % Normalize homogeneous coordinates
% Extract x and y coordinates of the 2D points
x_cube = cube_2d(1, :);
y_cube = cube_2d(2, :);
% Load and display the image
figure;
imshow(image);
hold on;
plot(x_cube,y_cube, 'yx', 'MarkerSize', 10, 'LineWidth', 2);  % Points transformés en jaune
% Draw the edges of the cube
% Define pairs of indices for the cube edges
%% dessin cube 
% Charger et afficher l'image
image = imread('APC_0016.jpg'); % Remplace par le chemin de ton image
imshow(image);
hold on; % Garder l'image affichée pour ajouter les éléments
;

% Extraire les coordonnées x et y
x_cube = cube_2d(1, :); % Coordonnées x
y_cube = cube_2d(2, :); % Coordonnées y

% Relier les points de la base inférieure (points 1 à 4)
plot([x_cube(1), x_cube(2)], [y_cube(1), y_cube(2)], 'm-', 'LineWidth', 2);
plot([x_cube(2), x_cube(4)], [y_cube(2), y_cube(3)], 'm-', 'LineWidth', 2);
plot([x_cube(3), x_cube(4)], [y_cube(3), y_cube(4)], 'm-', 'LineWidth', 2);
plot([x_cube(3), x_cube(1)], [y_cube(4), y_cube(1)], 'm-', 'LineWidth', 2);

% Relier les points de la base supérieure (points 5 à 8)
plot([x_cube(5), x_cube(6)], [y_cube(5), y_cube(6)], 'm-', 'LineWidth', 2);
plot([x_cube(6), x_cube(8)], [y_cube(6), y_cube(7)], 'm-', 'LineWidth', 2);
plot([x_cube(7), x_cube(8)], [y_cube(7), y_cube(8)], 'm-', 'LineWidth', 2);
plot([x_cube(7), x_cube(5)], [y_cube(8), y_cube(5)], 'm-', 'LineWidth', 2);

% Relier les coins entre la base inférieure et supérieure (points 1 à 5, 2 à 6, etc.)
plot([x_cube(1), x_cube(5)], [y_cube(1), y_cube(5)], 'm-', 'LineWidth', 2);
plot([x_cube(2), x_cube(6)], [y_cube(2), y_cube(6)], 'm-', 'LineWidth', 2);
plot([x_cube(3), x_cube(7)], [y_cube(3), y_cube(7)], 'm-', 'LineWidth', 2);
plot([x_cube(4), x_cube(8)], [y_cube(4), y_cube(8)], 'm-', 'LineWidth', 2);

% Ajouter les points du cube
plot(x_cube, y_cube, 'bo', 'MarkerFaceColor', 'g', 'MarkerSize', 8);

% Ajuster l'affichage pour ne pas cacher les éléments
hold off;

   











