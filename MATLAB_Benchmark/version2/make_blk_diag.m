function [blockdiag] = make_blk_diag(vector) 
    % Number of columns
    numCols = size(vector, 2);
    
    % Initialize an empty cell array
    Cell = cell(1, numCols);
    
    % Populate the cell array with individual columns
    for i = 1:numCols
        Cell{i} = vector(:, i);
    end
    
    % Create a block diagonal matrix using blkdiag
    blockdiag = blkdiag(Cell{:});
end