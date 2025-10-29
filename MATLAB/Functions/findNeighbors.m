function pairs = findNeighbors(distance, varargin)
% findNeighbors finds neighboring electrodes within a hardware piece
%
% Inputs:
%   distance - number of units away (ex: 2 finds 2-neighbors *only*)
%   
% 
% Optional Key-Value Input:
%   useDiagonal, true/[false] - if true, distance is on diagonal
%   elementInfo, info - can pass in elementInfo to avoid loading it (you
%                       can skip subj/rootEEGdir params in this case)
%   
% Output:
%   pairs - n by 2 array of neighbor pairs
%
% Revision History
%   7/17 MST - Updated
%
% See Also: createBipolarPairs

%  Copyright (C) 2017 Mike Trotta
%  Adjusted by Bloem van Dam to make it feasible for just one grid as
%  input, 2024

    % parse input
    ip = inputParser;
    ip.addParameter('useDiagonal', false);
    ip.addParameter('elementInfo', [], @istable);
    ip.parse(varargin{:});
    info = ip.Results.elementInfo;
    useDiagonal = ip.Results.useDiagonal;

    pairs = [];
    x_max = info.smallDim;
    y_max = info.bigDim;
    
   
    % set dx/dy to neighbors
    if useDiagonal
        step1 = distance * [1,1];
        step2 = distance * [1,-1];
    else
        step1 = distance * [0,1];
        step2 = distance * [1,0];
    end
    
    % row/col to index
    getNum = @(x,y) (x-1) * info.bigDim + y;
    
    % for each electrode, find 2 neighbors and add their edge
    for x = 1 : x_max
        for y = 1 : y_max
            
            n = getNum(x,y);
            if ismember(n, missing), continue; end
            
            x1 = x + step1(1);
            y1 = y + step1(2);
            
            x2 = x + step2(1);
            y2 = y + step2(2);
            
            n1 = getNum(x1, y1);
            n2 = getNum(x2, y2);
            
            if ~ismember(n1, missing) &&  (1 <= x1 && x1 <= x_max) && (1 <= y1 && y1 <= y_max)
                pairs = [pairs; n, n1];
            end
            
            if ~ismember(n2, missing) &&  (1 <= x2 && x2 <= x_max) && (1 <= y2 && y2 <= y_max)
                pairs = [pairs; [n, n2]];
            end
            
        end
    end
    
    
end
