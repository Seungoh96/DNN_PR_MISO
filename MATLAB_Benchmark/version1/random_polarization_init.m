function [Pol_Tx, Pol_Rx, Pol_Tx_blk, Pol_Rx_blk] = random_polarization_init(nt, nr)
    % Initialize Polarization vector storages
    Pol_Rx = zeros(2,nr);
    Pol_Tx = zeros(2,nt);

    % Initialization for random receiver angle
    for n_Rx = 1:nr
        theta_rand_Rx = rand * pi;
        Pol_Rx(1,n_Rx) = cos(theta_rand_Rx);
        Pol_Rx(2,n_Rx) = sin(theta_rand_Rx);
    end
    % Initialization for random transmitter angle
    for n_Tx = 1:nt
        theta_rand_Tx = rand * pi;
        Pol_Tx(1,n_Tx) = cos(theta_rand_Tx);
        Pol_Tx(2,n_Tx) = sin(theta_rand_Tx);
    end

    % Create block diagonal matrix using blkdiag
    Pol_Rx_blk = make_blk_diag(Pol_Rx);
    Pol_Tx_blk = make_blk_diag(Pol_Tx);
end