function [distances, stats] = validation_grid(grid_coor, gold_standard, anchor_coords)
% Function for validation of the grid compared to the gold standard
% Inputs:
% grid_coor     - Coordinates of the grid you want to compare to the gold
%               standard as [number, x, y, z]
% gold_standard - Coordinates of the gold standard as [number, x, y, z]
% anchor_coords - Coordinates of the anchors to ensure exclusion from
%               validation as [number, x, y, z]
% Outputs:
% distances     - Euclidean distances (accuracy) between every electrode of input grid
%               and corresponding gold standard electrode
% stats         - struct with the mean accuracy and standard deviation, 
%               median accuracy and interquartile range, and max distance 

if istable(grid_coor)
    grid_coor = table2array(grid_coor);
end

if size(grid_coor, 2) ~= 4
    grid_coor = [(1:64)', grid_coor];
end
% Initialize logical index for exclusion of anchors
exclude_anchor = ismember(grid_coor(:,1), anchor_coords(:,1));
grid_coor = grid_coor(~exclude_anchor,:);

% Only save results with corresponding gold standard electrodes
exclude_row = false(length(gold_standard), 1);
distances = zeros(length(gold_standard),1);
for i = 1: length(gold_standard)
    index = find(grid_coor(:,1) == gold_standard(i,1));
    if ~isempty(index)
        distance = pdist2(gold_standard(i, 2:4), grid_coor(index, 2:4), 'euclidean');
        distances(i, 2) = distance;
        distances(i, 1) = gold_standard(i,1);
    else
        % Exclude the entire row from results when no corresponding gold_standard entry is found
        exclude_row(i) = true;
    end
end
distances(exclude_row, :) = [];
distances_stats = distances(:,2); 

stats.mean_dist = mean(distances_stats);
stats.std_dist = std(distances_stats);
stats.median_dist = median(distances_stats);
stats.iqr = quantile(distances_stats,[0.25 0.75]);
stats.hausdorff = max(distances_stats);
end