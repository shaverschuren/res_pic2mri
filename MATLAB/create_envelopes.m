function create_envelopes(surf_dir)
    % CREATE_ENVELOPES Create brain envelopes from a pial surface stl files.
    %                  Creates lh/rh/whole brain envelopes.
    %                  Input surf_dir is the path to a freesurfer /surf dir
    %                  Author: Sjors Verschuren, 14-10-2025
    %                          Adjusted from enveloppe_freesurfer.m &
    %                          main_3Dcomponents.m in LD-ioECoG_to_preopMRI

    % Set filenames
    input_lh_stl = surf_dir + "\lh_pial.stl";
    input_rh_stl = surf_dir + "\rh_pial.stl";
    lh_envelope_stl = surf_dir + "\lh_envelope.stl";
    rh_envelope_stl = surf_dir + "\rh_envelope.stl";
    brain_envelope_stl = surf_dir + "\brain_envelope.stl";

    % Load files
    fv_lh = stlread(input_lh_stl);         % Load the lh STL file
    vertices_lh = fv_lh.Points;               % Extract vertices
    faces_lh = fv_lh.ConnectivityList;        % Extract faces

    fv_rh = stlread(input_rh_stl);         % Load the rh STL file
    vertices_rh = fv_rh.Points;               % Extract vertices
    faces_rh = fv_rh.ConnectivityList;        % Extract faces

    disp("Loaded .stl files")

    %% Create masks
    disp("Computing masks...")

    % Resolution of the mask (voxel size)
    voxel_size = 1;  % in mm
    
    % Expand the bounding box slightly to avoid cutoff
    margin_voxels = 5;   % expand by 5 voxels in all directions
    margin = margin_voxels * voxel_size;

    % Define the bounding box for lh with margin
    x_min_lh = floor(min(vertices_lh(:, 1))) - margin;
    x_max_lh = ceil(max(vertices_lh(:, 1))) + margin;
    y_min_lh = floor(min(vertices_lh(:, 2))) - margin;
    y_max_lh = ceil(max(vertices_lh(:, 2))) + margin;
    z_min_lh = floor(min(vertices_lh(:, 3))) - margin;
    z_max_lh = ceil(max(vertices_lh(:, 3))) + margin;

    % Define the bounding box for rh with margin
    x_min_rh = floor(min(vertices_rh(:, 1))) - margin;
    x_max_rh = ceil(max(vertices_rh(:, 1))) + margin;
    y_min_rh = floor(min(vertices_rh(:, 2))) - margin;
    y_max_rh = ceil(max(vertices_rh(:, 2))) + margin;
    z_min_rh = floor(min(vertices_rh(:, 3))) - margin;
    z_max_rh = ceil(max(vertices_rh(:, 3))) + margin;

    % Combine bounding boxes for overall envelope
    x_min = min(x_min_lh, x_min_rh);
    x_max = max(x_max_lh, x_max_rh);
    y_min = min(y_min_lh, y_min_rh);
    y_max = max(y_max_lh, y_max_rh);
    z_min = min(z_min_lh, z_min_rh);
    z_max = max(z_max_lh, z_max_rh);
    
    % Create separate 3D grids for lh and rh masks for speed
    [x_lh, y_lh, z_lh] = meshgrid(x_min_lh:voxel_size:x_max_lh, ...
                                  y_min_lh:voxel_size:y_max_lh, ...
                                  z_min_lh:voxel_size:z_max_lh);

    [x_rh, y_rh, z_rh] = meshgrid(x_min_rh:voxel_size:x_max_rh, ...
                                  y_min_rh:voxel_size:y_max_rh, ...
                                  z_min_rh:voxel_size:z_max_rh);

    % Create overall grid for envelope
    [x, y, z] = meshgrid(x_min:voxel_size:x_max, ...
                         y_min:voxel_size:y_max, ...
                         z_min:voxel_size:z_max);

    % Check which grid points lie inside the lh mesh using its own grid
    mask_lh = inpolyhedron(faces_lh, vertices_lh, [x_lh(:), y_lh(:), z_lh(:)]);
    mask_lh = reshape(mask_lh, size(x_lh));

    % Check which grid points lie inside the rh mesh using its own grid
    mask_rh = inpolyhedron(faces_rh, vertices_rh, [x_rh(:), y_rh(:), z_rh(:)]);
    mask_rh = reshape(mask_rh, size(x_rh));

    % volshow(mask); % For MATLAB R2020a or later
    
    %% Perform dilation/smoothing for lh/rh masks
    disp("Performing dilation/smoothing on separate hemis...")

    se = strel('sphere', 3);  % Structuring element with radius 3
    lh_dilated = imdilate(mask_lh, se);
    rh_dilated = imdilate(mask_rh, se);
    
    lh_smoothed = imgaussfilt3(double(lh_dilated), 7.0);
    rh_smoothed = imgaussfilt3(double(rh_dilated), 7.0);

    %% Get iso surface

    fv_iso_lh = isosurface(lh_smoothed, 0.5);  % Extract isosurface at threshold 0.5
    fv_iso_rh = isosurface(rh_smoothed, 0.5);  % Extract isosurface at threshold 0.5

    %% Place lh and rh masks back into the overall envelope grid
    mask = false(size(x));
    % Compute indices for placing lh mask
    x_idx_lh = (x_lh(:) - x_min) / voxel_size + 1;
    y_idx_lh = (y_lh(:) - y_min) / voxel_size + 1;
    z_idx_lh = (z_lh(:) - z_min) / voxel_size + 1;
    ind_lh = sub2ind(size(mask), y_idx_lh, x_idx_lh, z_idx_lh);
    mask(ind_lh) = mask_lh(:);

    % Compute indices for placing rh mask
    x_idx_rh = (x_rh(:) - x_min) / voxel_size + 1;
    y_idx_rh = (y_rh(:) - y_min) / voxel_size + 1;
    z_idx_rh = (z_rh(:) - z_min) / voxel_size + 1;
    ind_rh = sub2ind(size(mask), y_idx_rh, x_idx_rh, z_idx_rh);
    mask(ind_rh) = mask(ind_rh) | mask_rh(:);

    % Treshold
    mask = mask > 0.5;

    %% Perform closing on whole brain envelope
    disp("Performing closing/smoothing on whole brain...")

    se = strel('sphere', 15);
    wb_closed = imclose(mask, se);
    wb_dilated = imdilate(wb_closed, strel('sphere', 3));
    wb_smoothed = imgaussfilt3(double(wb_dilated), 7.0);

    fv_iso_wb = isosurface(wb_smoothed, 0.5);

    %% Map voxel indices (from isosurface) back to original coordinates
    
    fv_iso_lh.vertices = fv_iso_lh.vertices * voxel_size + [x_min_lh, y_min_lh, z_min_lh];
    fv_iso_rh.vertices = fv_iso_rh.vertices * voxel_size + [x_min_rh, y_min_rh, z_min_rh];
    fv_iso_wb.vertices = fv_iso_wb.vertices * voxel_size + [x_min, y_min, z_min];

    %% Save envelopes as stl
    stlwrite(lh_envelope_stl, fv_iso_lh.faces, fv_iso_lh.vertices);
    stlwrite(rh_envelope_stl, fv_iso_rh.faces, fv_iso_rh.vertices);
    stlwrite(brain_envelope_stl, fv_iso_wb.faces, fv_iso_wb.vertices);

end