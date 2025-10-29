function grid_coor = anchortogrid(anchor_coords,GridHeight, GridWidth, elec_spacing, showFigure)
% Author: Bloem van Dam
% Date: 13-2-2024
% This function allows you to match the manually selected anchor points with the MRI used
% for segmentation and inter/extrapolate the other grid points, making a rigid
% grid representation in RAS orientation in mm space
%
% Inputs:
% anchor_coords - coordinates of the anchor points as [number, x, y, z]
%
% Output:
% grid_coor     - Coordinates of a rigid grid in mm space in RAS+ orientation based on
%                 brainlab coordinates
%% Transformation of anchor electrodes to rigid grid
anchor_points= anchor_coords(:,2:4);
electrodes = anchor_coords(:,1);
% Co-registration of electrode grid
%define grid in  mm domain

if nargin == 1
    GridHeight = 8;         % Number of electrodes in one row 
    GridWidth = 8;          % Number of electrodes in one column
    elec_spacing = 5;       % Electrode spacing in mm    
end

% Generate grid points
[X, Y] = meshgrid((GridWidth-1:-1:0)*elec_spacing, (GridHeight-1:-1:0)*elec_spacing);
Z = zeros(size(X));
Grid_templ = [X(:), Y(:), Z(:)];
% Registered electrodes in Slicer as anchor on grid template 
elec_reg = Grid_templ(electrodes,:);
% Transform grid template to obtained brainlab points in RAS+ mm space
[~ ,~ , transform] = procrustes(anchor_points, elec_reg,'scaling',false, 'reflection' , false);

% Perform transformations on grid and visualize all electrodes
grid_coor = [];
for i = 1:(GridHeight*GridWidth)
    grid_coor = [grid_coor; Grid_templ(i, 1:3)*transform.T+transform.c(1,:)]; % In mm space
end

if showFigure
    figure  % For visual inspection
    hold on
    plot3(Grid_templ(:,1),Grid_templ(:,2),Grid_templ(:,3),'b*');
    plot3(anchor_points(:,1),anchor_points(:,2),anchor_points(:,3),'r*');
    plot3(grid_coor(:,1),grid_coor(:,2),grid_coor(:,3),'g*')
    hold off
end
end