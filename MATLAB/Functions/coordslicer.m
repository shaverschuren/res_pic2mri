function markups_slicer = coordslicer_MNI(path, situation)
% Function to read the coordinates saved in Slicer and load them in RAS+
% orientation
% Inputs:
% path              - path to folder with markup files from 3D Slicer (in LPS+ orientation)
% situation         - the number and character for the grid placement (e.g. '1A')

% Outputs:
% markups_slicer    - The markups in slicer as [number, x, y, z] in RAS+
%                   orientation

% Read the JSON path
jsonFiles = dir(fullfile(path, ['*' situation '*']));

% Initialize the variable to store positions
markups_slicer = zeros(length(jsonFiles), 4);

for i = 1:length(jsonFiles)
    % Read the JSON file
    jsonFilePath = fullfile(path, jsonFiles(i).name);
    jsonText = fileread(jsonFilePath);
    
    % Parse the JSON content
    jsonData = jsondecode(jsonText);
    
    % Extract the position and electrode number
    position = jsonData.markups(1).controlPoints(1).position;
    electrodeNumber = str2double(regexp(jsonFiles(i).name, '\d+', 'match'));
    
    % Store the information in the variable
    markups_slicer(i, 2:4) = position;
    markups_slicer(i, 1) = electrodeNumber(2);
end

C = [1 0 0; 0 1 0 ; 0 0 1];
markups_temp = zeros(size(markups_slicer,1), 3);
for j = 1:size(markups_slicer,1)
    markups_temp(j,:) = (C*markups_slicer(j,2:4)')';
    markups_slicer(j,:) = [markups_slicer(j,1), markups_temp(j,:)];
end
end