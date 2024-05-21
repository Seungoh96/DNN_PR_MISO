clear 
clc 

%% Simulation parameters
Joint_prepost_iteration = 10;   % number of iteration for polarization optimization
niteration = 1;              % number of iteration for MC simulation
Pt_dBm = 0;                     % dowlink power in dBm
N0_dBm = 0;                     % noise power in dBm
Pt = 10^(Pt_dBm/10);            % downlink power in linear
N0 = 10^(N0_dBm/10);            % downlink noise in linear
nt = 64;                        % number of transmitters
nr = 1;                         % number of receivers
SNR = 10*log10(Pt/N0);          % raw SNR in linear scale
L = 2;                          % total number of pilot transmission
disp('SNR = ')                  % display raw SNR
disp(SNR)

% Initialization
C_rand_randW = zeros(1,niteration);
C_rand_optW = zeros(1,niteration);
C_opt_randW = zeros(1,niteration);
C_opt_optW = zeros(1, niteration);
C_hat_randW = zeros(1, niteration);
C_hat_optW = zeros(1, niteration);
mseUE = zeros(1, niteration); 
mseBS = zeros(1, niteration);
%% Getting Channel 
for i = 1:niteration
    disp("iteration = ")
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
    [Pol_Tx_opt, Pol_Rx_opt, Pol_Tx_opt_blk, Pol_Rx_opt_blk] = Pol_JointPrePostCoding_test(H, nt, nr, Joint_prepost_iteration);
    [Pol_BS_opt_MC, Pol_UE_opt_MC, Pol_BS_opt_MC_blk, Pol_UE_opt_MC_blk] = MC_Optimization(H, nt, nr, Joint_prepost_iteration);
    Heff_opt = Pol_Rx_opt_blk' * H * Pol_Tx_opt_blk;
    Heff_MC = Pol_UE_opt_MC_blk' * H * Pol_BS_opt_MC_blk;
    W_opt_rand = get_beamformer(Heff_opt, Pt, "random");
    W_opt_opt = get_beamformer(Heff_opt, Pt, "optimal");
    W_opt_opt_MC = get_beamformer(Heff_MC, Pt, "optimal");
    C_opt_randW(1,i) = calculate_capacity(Heff_opt, W_opt_rand, N0);
    C_opt_optW(1,i) = calculate_capacity(Heff_opt, W_opt_opt, N0);
    C_opt_optW_MC(1,i) = calculate_capacity(Heff_MC, W_opt_opt_MC, N0);
    
    %% ------------------------------------------------------------------------
    
    %% Imperfect CSI
    %% ------------------------Pilot Generation--------------------------------
    [Pol_BS_t_pilot, Pol_BS_r_pilot, Pol_UE_t_pilot, Pol_UE_r_pilot, W_pilot] = ...
        get_pilot_parameters(L, nt, nr, Pt);
    [Y_BS, Y_UE, Pol_BS_t_blk_pilot, Pol_BS_r_blk_pilot_full, Pol_UE_t_blk_pilot, Pol_UE_r_blk_pilot, W] = ...
        generate_pilot(H,Pol_BS_t_pilot,Pol_BS_r_pilot,Pol_UE_t_pilot,Pol_UE_r_pilot,W_pilot,L,N0);
    
    %% ------------------------Channel Estimation------------------------------
    % BS Side 
    Y_BS = reshape(Y_BS, [], 1);
    A = reshape(Pol_BS_r_blk_pilot_full, [2*nt,nt*L]);
    A = A';
    H_BS_hat = Channel_Estimation_BS(Y_BS, A, Pol_UE_t_blk_pilot);
    H_BS_hat = transpose(H_BS_hat);
    
    % UE Side 
    A = transpose(Pol_BS_t_blk_pilot*W);
    Y_UE = transpose(Y_UE);
    H_UE_hat = Channel_Estimation_UE(Y_UE, A, Pol_UE_r_blk_pilot);
    H_UE_hat = transpose(H_UE_hat);

    %% ----------------------Apply Algorithm for Polarization------------------
    % BS Side 
    [Pol_BS_Tx_opt, Pol_BS_Rx_opt, Pol_BS_Tx_opt_blk, Pol_BS_Rx_opt_blk] = Pol_JointPrePostCoding(H_BS_hat, nt, nr, Joint_prepost_iteration);
    Heff_BS_hat = Pol_BS_Rx_opt_blk' * H_BS_hat * Pol_BS_Tx_opt_blk;
    W_hat_rand = get_beamformer(Heff_BS_hat, Pt, "random");
    W_hat_opt = get_beamformer(Heff_BS_hat, Pt, "optimal");
    
    % UE Side 
    [~, Pol_UE_Rx_opt, ~, Pol_UE_Rx_opt_blk] = Pol_JointPrePostCoding(H_UE_hat, nt, nr, Joint_prepost_iteration);
    
    % Getting Rate 
    Heff_hat = Pol_UE_Rx_opt_blk' * H * Pol_BS_Tx_opt_blk;
    
    C_hat_randW(1,i) = calculate_capacity(Heff_hat, W_hat_rand, N0);
    C_hat_optW(1,i) = calculate_capacity(Heff_hat, W_hat_opt, N0);

    %% ----------------------Estimation Performance------------------------
    mseUE(1,i) = norm(H - H_UE_hat, 'fro')/norm(H,'fro');
    mseBS(1,i) = norm(H - H_BS_hat, 'fro')/norm(H,'fro');
end

C_random_randomW = mean(C_rand_randW)
C_random_optW = mean(C_rand_optW)

C_optimal_randW = mean(C_opt_randW)
C_optimal_optW = mean(C_opt_optW)
C_optimal_optW_MC = mean(C_opt_optW_MC)

C_hat_randomW = mean(C_hat_randW)
C_hat_optimalW = mean(C_hat_optW)

mmseBS = mean(mseUE)
mmseUE = mean(mseBS)





