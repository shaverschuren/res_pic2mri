function neighborMask = getNeighborMask(elementInfo, myCoords, saveFile)
    % pairs/edges between close electrodes are found:
    %  + shear/bend/norm neighbors are considered pairs and found based on grid
    %    geometry.
    %  + inter-hardware edges are formed between electrodes within 15 mm
    %
    % Input:
    %   elementInfo - table with each row a hardware piece with columns colums:
    %       bigDim  - largest dimension (>1 for grids, 1 for strips)
    %       smallDim- other dimension
    %   myCoords    - table with x, y, z, chanName columns
    %   saveFile (optional) - where to save edges variable
    %
    % Output: 
    %   - neighborMask is a somewhat more compacted form of the usable data.
    %       x=squareform(configVec) gives the usable, n by n data, such that x(i,j)
    %       is 1 if there is an edge between electrodes i and j and 0 otherwise.
    %       Without calling squareform, you can also use this as a mask for the
    %       vector output from the pdist function.

%  Copyright (C) 2017 Mike Trotta
%  Adjusted by Bloem van Dam to make it feasible for just one grid as
%  input, 2024

    % element based connectivity:
    % note the edges include duplicates
    nchan = height(myCoords);
    myCoords.chanNdx = [1 : nchan]';
    edges = [[] []];


    % intra-hardware connectivity
    norm = findNeighbors(1, 'elementInfo',elementInfo);
    bend = findNeighbors(2, 'elementInfo',elementInfo);
    shear= findNeighbors(1, 'elementInfo',elementInfo, 'useDiagonal',true);
    intra_edges = [norm; bend; shear];


    edges = [edges; intra_edges];

    % connect edges with upper triangular adjacency matrix vector 
    amat = zeros(nchan, nchan);
    if ~isempty(edges)
        for iEdge = 1:length(edges(:,1))
            amat(edges(iEdge,1), edges(iEdge,2)) = 1;
            amat(edges(iEdge,2), edges(iEdge,1)) = 1;
        end
    end
    neighborMask = squareform(amat,'tovector');

    % save edges to file
    if exist('saveFile','var') && ~isempty(saveFile)
        save(saveFile, 'edges');
    end

end
