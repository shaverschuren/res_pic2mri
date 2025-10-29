function gii2stl(filename,switchLPS,filename_store)
% converts the gii output of cat12 to stl file format

% Nicole van Klink, 3-7-2019


if nargin <1
[file,path] = uigetfile('*.gii','Select gii file to import');
filename = [path file];
end

s = gifti(filename);

%ask for conversion to LPS (to read in EpiNav)
if nargin <2
switchLPS = questdlg('Do you want to convert to LPS coordinates (for EpiNav)?', ...
	'Do you want to convert to LPS coordinates (for EpiNav)?', ...
	'Yes','No','No');
end

switch switchLPS 
    case 'Yes'
    matrix = [-1 0 0; 0 -1 0; 0 0 1];
    s.vertices = s.vertices*matrix;
    disp('Transformed to LPS coordinates');
    case 'No'
    disp('No transformation to LPS coordinates');
end
    
%adjust filename
if nargin <3
    filename(end-2:end) = 'stl';
    stlwrite(filename,s.faces, s.vertices); 
else
    stlwrite(filename_store,s.faces, s.vertices);
end
