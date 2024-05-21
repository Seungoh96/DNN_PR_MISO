clear 
clc 

%% Simulation parameters
Joint_prepost_iteration = 10;   % number of iteration for polarization optimization
niteration = 1;
Pt_dBm = 0;                     % dowlink power in dBm
N0_dBm = 0;                     % noise power in dBm
Pt = 10^(Pt_dBm/10);            % downlink power in linear
N0 = 10^(N0_dBm/10);            % downlink noise in linear
nt = 4;                        % number of transmitters
nr = 1;                         % number of receivers
SNR = 10*log10(Pt/N0);          % raw SNR in linear scale
L = 2;                          % total number of pilot transmission
disp('SNR = ')                  % display raw SNR
disp(SNR)

%% Getting Channel 
for i = 1:niteration
    disp(i)
    H_p_real = random('Normal',0,1,2*nr,2*nt);  % real part 
    H_p_imag = random('Normal',0,1,2*nr,2*nt);  % imaginary part
    H = (H_p_real + 1i*H_p_imag) / sqrt(2);     % normalizing 
    
    %% Perfect CSI
    %% -------------------------Random Polarization----------------------------
    [Pol_Tx_rand, Pol_Rx_rand, Pol_Tx_rand_blk, Pol_Rx_rand_blk] = random_polarization_init(nt, nr);
    Heff_rand = Pol_Rx_rand_blk' * H * Pol_Tx_rand_blk;
    W_rand_rand = get_beamformer(Heff_rand, Pt, "random");
    W_rand_opt = get_beamformer(Heff_rand, Pt, "optimal");
    C_rand_randW(1,i) = calculate_capacity(Heff_rand, W_rand_rand, N0);
    C_rand_optW(1,i) = calculate_capacity(Heff_rand, W_rand_opt, N0);
    
    %% -------------------------Optimal Polarization---------------------------
    [Pol_Tx_opt, Pol_Rx_opt, Pol_Tx_opt_blk, Pol_Rx_opt_blk] = Pol_JointPrePostCoding(H, nt, nr, Joint_prepost_iteration);
    Heff_opt = Pol_Rx_opt_blk' * H * Pol_Tx_opt_blk;
    W_opt_rand = get_beamformer(Heff_opt, Pt, "random");
    W_opt_opt = get_beamformer(Heff_opt, Pt, "optimal");
    C_opt_randW(1,i) = calculate_capacity(Heff_opt, W_opt_rand, N0);
    C_opt_optW(1,i) = calculate_capacity(Heff_opt, W_opt_opt, N0);
    
    %% ------------------------------------------------------------------------
    
    %% Imperfect CSI
    %% ------------------------Pilot Generation--------------------------------
    mode = ["random", "LS_Method" ,"tpi4_rVH", "tVH_rVH"];
    [Pol_BS_t_pilot, Pol_BS_r_pilot, Pol_UE_t_pilot, Pol_UE_r_pilot, W_pilot] = ...
        get_pilot_parameters(L, nt, nr, Pt, mode(1));
    [Y_BS, Y_UE, Pol_BS_t_blk_pilot, Pol_BS_r_blk_pilot, Pol_UE_t_blk_pilot, Pol_UE_r_blk_pilot] = ...
        generate_pilot(H,Pol_BS_t_pilot,Pol_BS_r_pilot,Pol_UE_t_pilot,Pol_UE_r_pilot,W_pilot,L,N0,mode(1));
    
    %% ------------------------Channel Estimation------------------------------
    % Get necessary blocks for estimation
    [H_block,H_T_block,Pol_BS_r_blk_pilot_block,Pol_UE_t_blk_pilot_block,Pol_UE_r_blk_pilot_block,BS_tW_block] = ...
        get_block_parameters(H, W_pilot,Pol_BS_t_blk_pilot,Pol_BS_r_blk_pilot,Pol_UE_t_blk_pilot,Pol_UE_r_blk_pilot);
    
    % Perform estimation protocol
    estimation_method = ["LS", "LMMSE"];
    [H_BS_hat_combined, H_UE_hat_combined] = ...
        Channel_Estimation(Y_BS, Y_UE, Pol_UE_t_blk_pilot_block, BS_tW_block, estimation_method(1));
    
    %% ----------------------Apply Algorithm for Polarization------------------
    % BS Side 
    [Pol_BS_Tx_opt, Pol_BS_Rx_opt, Pol_BS_Tx_opt_blk, Pol_BS_Rx_opt_blk] = Pol_JointPrePostCoding(H_BS_hat_combined, nt, nr, Joint_prepost_iteration);
    Heff_BS_hat = Pol_BS_Rx_opt_blk' * H_BS_hat_combined * Pol_BS_Tx_opt_blk;
    W_hat_rand = get_beamformer(Heff_BS_hat, Pt, "random");
    W_hat_opt = get_beamformer(Heff_BS_hat, Pt, "optimal");
    
    % UE Side 
    [~, Pol_UE_Rx_opt, ~, Pol_UE_Rx_opt_blk] = Pol_JointPrePostCoding(H_UE_hat_combined, nt, nr, Joint_prepost_iteration);
    
    % Getting Rate 
    Heff_hat = Pol_UE_Rx_opt_blk' * H * Pol_BS_Tx_opt_blk;
    
    C_hat_randW(1,i) = calculate_capacity(Heff_hat, W_hat_rand, N0);
    C_hat_optW(1,i) = calculate_capacity(Heff_hat, W_hat_opt, N0);
end

C_random_randomW = mean(C_rand_randW)
C_random_optW = mean(C_rand_optW)

C_optimal_randW = mean(C_opt_randW)
C_optimal_optW = mean(C_opt_optW)

C_hat_randomW = mean(C_hat_randW)
C_hat_optimalW = mean(C_hat_optW)

mmseBS = norm(H - H_BS_hat_combined, 'fro')
mmseUE = norm(H - H_UE_hat_combined, 'fro')






