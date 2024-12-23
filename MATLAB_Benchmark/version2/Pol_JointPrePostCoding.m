function [Pol_Tx_Opt, Pol_Rx_Opt, Pol_Tx_Opt_blk, Pol_Rx_Opt_blk] = Pol_JointPrePostCoding(H, Nt, Nr, N_iter)
%% Initialization 
    % Initiate vectors to store parameters 
    Pol_Rx_Opt = zeros(2,Nr);       % stores the final Pol_Rx
    Pol_Tx_Opt = zeros(2,Nt);       % stores the final Pol_Tx
    theta_Tx = zeros(Nt,1);         % stores the optimal Tx polarization angles
    theta_Rx = zeros(Nr,1);         % stores the optimal Rx polarization angles
    Pol_Rx_Opt_temp = zeros(2,1);   % used during joint prepostcoding
    Pol_Tx_Opt_temp = zeros(2,1);   % used during joint prepostcoding

    %Initialization of Opt Rx-Pol. vectors
    for n_Rx = 1 : Nr
        theta_rand_Rx = rand * (pi);
        Pol_Rx_Opt(1,n_Rx) = cos(theta_rand_Rx);    Pol_Rx_Opt(2,n_Rx) = sin(theta_rand_Rx);
    end
    
    %Initialization of Opt Tx-Pol. vectors
    for n_Tx = 1 : Nt     
        theta_rand_Tx = rand * (pi);
        Pol_Tx_Opt(1,n_Tx) = cos(theta_rand_Tx);    Pol_Tx_Opt(2,n_Tx) = sin(theta_rand_Tx);
    end 
    
    index_Tx = 1:2:2*Nt;
    index_Rx = 1:2:2*Nr;
%% Joint PrePostCoding
    for n_prepost = 1:N_iter    % number of joint iteration for optimization
        % ------- Tx Optimization ---------
        H_temp = zeros(2,2);
        for n_Tx = 1:Nt
            Tx_index = index_Tx(n_Tx);
            H_PD_Tx_temp = zeros(2,2);
            for n_Rx = 1:Nr
                Rx_index = index_Rx(n_Rx);
                H_temp(:,:) = H(Rx_index:Rx_index+1,Tx_index:Tx_index+1);
                Pol_Rx_Opt_temp(:) = Pol_Rx_Opt(:,n_Rx);
                H_PD_Tx_temp = H_PD_Tx_temp + H_temp'*Pol_Rx_Opt_temp*(Pol_Rx_Opt_temp)'*H_temp;
            end
            [S,A] = eig( (H_PD_Tx_temp) );      
            A_PD_Vec = diag(A);
            [~, I_lamda_PD_max] = max(A_PD_Vec);
            theta_Tx(n_Tx) = angle( S(1,I_lamda_PD_max) + 1i * S(2,I_lamda_PD_max) );
            Pol_Tx_Opt(1,n_Tx) = cos(theta_Tx(n_Tx));    Pol_Tx_Opt(2,n_Tx) = sin(theta_Tx(n_Tx));
        end
        % -------- Rx Optimization---------
        for n_Rx = 1:Nr
            Rx_index = index_Rx(n_Rx);
            H_PD_Rx_temp = zeros(2,2);
            for n_Tx = 1:Nt
                Tx_index = index_Tx(n_Tx);
                H_temp(:,:) = H(Rx_index:Rx_index+1,Tx_index:Tx_index+1);
                Pol_Tx_Opt_temp(:) = Pol_Tx_Opt(:,n_Tx);
                H_PD_Rx_temp = H_PD_Rx_temp + H_temp*Pol_Tx_Opt_temp*(Pol_Tx_Opt_temp)'*H_temp';
            end
            [S,A] = eig( (H_PD_Rx_temp) );      
            A_PD_Vec = diag(A);
            [~,I_lamda_PD_max] = max(A_PD_Vec);
            theta_Rx(n_Rx) = angle( S(1,I_lamda_PD_max) + 1i * S(2,I_lamda_PD_max) );
            Pol_Rx_Opt(1,n_Rx) = cos(theta_Rx(n_Rx));    Pol_Rx_Opt(2,n_Rx) = sin(theta_Rx(n_Rx));
        end
    end
%% Make Block Diagonal Matrix 
    Pol_Rx_Opt_blk = make_blk_diag(Pol_Rx_Opt);
    Pol_Tx_Opt_blk = make_blk_diag(Pol_Tx_Opt);
end