function [proj_coords, dural_coords, dural_dist] = projection(rigid_grid, envelope_surf, anchor_coords, element_info,GridHeigth, GridWidth ,varargin)
% projection 
%
% Usage: proj_coords = PROJECTION(mr_coords, envelope_surf, anchor_coords, element_info
%
% Inputs:
%   rigid_grid      - array of xyz-coordinates of electrodes in rigid grid
%   envelope_surf   - A struct with faces and vertices fields representing the cortical envelope mesh
%   anchor_coords   - array of anchor xyz-coordinates for subset of electrodes in pre-operative MRI space.
%   element_info    - A table which describes the layout / geometry of strip/grid hardware
%                     Each row is a hardware piece. And columns contain at least the following fields
%                     - bigDim (numeric, use 1 for strips)
%                     - smallDim (numeric)                  
% Optional Input:
%
% Input key-values:
%   log_file        - path to log file. 
%   
% Output: 
%   proj_coords     - table of projected coordinate locations. This table will be of the same form as
%                     the mr_coords table
%   dural_coords    - same as proj_coords, but coords have been snapped to nearest point on dural
%   dural_dist      - table in the same form as mr_coords with a dist2dural column giving distance in mm to nearest point on dural
%
% File Output:
%        
%
% Description:
%   Given a set of XYZ coordinates of electrodes, a  cortical envelope surface, 
%   and coordinates of anchor points, solves an spring-model to project electrodes to the 
%   cortical envelope. Use this for one hemisphere at a time.
%   

% Revision History:
%   03/17 - MST
                                  
%  Copyright (C) 2017 Mike Trotta. Edited 2024 by Bloem van Dam
    
    % -------------
    %%--- Setup ---
    % -------------
    % Input Bloem

    
    mr_coords = [(1:(GridHeigth*GridWidth))' rigid_grid];
    mr_coords = array2table(mr_coords, 'VariableNames',{'chanName', 'x', 'y', 'z'});
    anchor_coords = array2table(anchor_coords,'VariableNames', {'chanName', 'x', 'y', 'z'}); 

    k_params.e_disp_weight  = 1;    % 1    - distance to original rigid grid coordinates
    k_params.e_fit_weight   = 25;   % 25   - distance to dural surface
    k_params.e_anch_weight  = 200;  % 200  - distance to anchor coordinate
    k_params.e_def_weight   = 1000; % 1000 - change in inter-electrode distances from CT coordinates
    
    % Input
    ip = inputParser;
    ip.addOptional('working_dir', pwd, @ischar);
    ip.addParameter('log_file', []);
    ip.parse(varargin{:});
    log_file = ip.Results.log_file;
    
    % Check input
    required_cols = {'chanName', 'x','y','z'};
    assert(istable(mr_coords) && length(required_cols) == length(intersect(required_cols, mr_coords.Properties.VariableNames)),...
        'mr_coords must be a table with chanName, x, y, and z columns');
    if ~isempty(anchor_coords)
        assert(istable(anchor_coords) && length(required_cols) == length(intersect(required_cols, anchor_coords.Properties.VariableNames)),...
            'anchor_coords must be a table with chanName, x, y, and z columns');
    end

    required_fields = {'faces','vertices'};
    if sum(ismember(required_fields, fieldnames(envelope_surf))) < 2
        error('Pial and evelope surfaces must be structure with .faces and .vertices fields');
    end
    duralSurfVerts = envelope_surf.vertices;
    
    % tables to mats
    if isempty(anchor_coords)
        anchor_ndx_xyz = [];
        coords_ndx_xyz  = cat(2, [1:height(mr_coords)]', mr_coords{:,{'x','y','z'}});
    else
        uncommon_anchors = setdiff(anchor_coords.chanName, mr_coords.chanName);
        if ~isempty(uncommon_anchors)
            warning('The following anchors were not found in mr_coords: %s', strjoin(uncommon_anchors, ' '));
        end
        anchor_coords   = anchor_coords(~ismember(anchor_coords.chanName, uncommon_anchors), :);
        % these next few lines ensure anchor ndx matches that in mr_coords
        [~,temp] = ismember(mr_coords.chanName, anchor_coords.chanName);
        [anchor_ndx,~,sort_ndx] = find(temp);
        anchor_xyz      = anchor_coords{sort_ndx,{'x','y','z'}};
        coords_ndx_xyz  = cat(2, [1:height(mr_coords)]', mr_coords{:,{'x','y','z'}});
        anchor_ndx_xyz  = cat(2, anchor_ndx, anchor_xyz);
    end
    
    neighbor_mask = getNeighborMask(element_info, mr_coords);
    
    % ----------------------
    %%--- Run Projection ---
    % ----------------------
    
    [ndx_xyz_dural, ndx_xyz_nosnap, ndx_d_dural] = projection_wrapper(coords_ndx_xyz, anchor_ndx_xyz, duralSurfVerts, k_params, neighbor_mask);
    proj_coords = mr_coords;
    proj_coords{:,{'x','y','z'}} = ndx_xyz_nosnap(:,2:4); % output of algorithm (without any dural snap constraint!)
    dural_coords = mr_coords;
    dural_coords{:,{'x','y','z'}} = ndx_xyz_dural(:,2:4);
    dural_dist = mr_coords(:,'chanName');
    dural_dist{:,{'dist2dural'}} = ndx_d_dural(:,2);
end
