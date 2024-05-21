function [blockdiag] = make_blk_diag_matrix(matrix) 
    % Number of columns
    numChannels = size(matrix, 3);
    
    % Initialize an empty cell array
    Cell = cell(1, numChannels);
    
    % Populate the cell array with individual columns
    for i = 1:numChannels
        Cell{i} = matrix(:, :, i);
    end
    
    % Create a block diagonal matrix using blkdiag
    blockdiag = blkdiag(Cell{:});
end