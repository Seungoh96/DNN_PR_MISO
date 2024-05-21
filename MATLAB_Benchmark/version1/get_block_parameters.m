function [H_block, H_T_block, Pol_BS_r_blk_pilot_block, Pol_UE_t_blk_pilot_block, ...
    Pol_UE_r_blk_pilot_block, BS_tW_block ] = ...
    get_block_parameters(H, W_pilot, Pol_BS_t_blk_pilot, Pol_BS_r_blk_pilot, ...
                            Pol_UE_t_blk_pilot, Pol_UE_r_blk_pilot)
    % Initialize 
    L = size(Pol_BS_t_blk_pilot,3);
    nt = size(Pol_BS_t_blk_pilot,2);
    TxW_block_tmp = zeros(2*nt,1,L);

    % UE Side 
    % Get reshaped Pol_Rx_block
    Pol_UE_r_blk_pilot_block = reshape(Pol_UE_r_blk_pilot, [1,2*L]);
    % Get TxW_block
    for i = 1:L
        TxW_block_tmp(:,:,i) = Pol_BS_t_blk_pilot(:,:,i)*W_pilot(:,:,i);
    end
    BS_tW_block = make_blk_diag_matrix(TxW_block_tmp);

    % BS Side 
    % Get reshaped Pol_BS_t_block
    Pol_BS_r_blk_pilot_block = reshape((permute(Pol_BS_r_blk_pilot, [2 1 3])), [nt,2*L*nt]);
    % Get reshaped Pol_UE_r_block
    Pol_UE_t_blk_pilot_block = make_blk_diag_matrix(Pol_UE_t_blk_pilot);

    % Get H_block 
    H_block = make_blk_diag_matrix(repmat(H,[1,1,L]));
    % Get H_T block 
    H_T = transpose(H);
    H_T_block = make_blk_diag_matrix(repmat(H_T,[1,1,L]));
end