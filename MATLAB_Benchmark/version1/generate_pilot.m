function [Y_BS,Y_UE,Pol_BS_t_blk_full,Pol_BS_r_blk_full,Pol_UE_t_blk_full,Pol_UE_r_blk_full] = ...
    generate_pilot(H, Pol_BS_t, Pol_BS_r, Pol_UE_t, Pol_UE_r, bf, L, N0, mode)
    [nr, nt] = size(H);             % getting system parameters
    nr = nr/2; nt = nt/2;           % getting system parameters
    Y_BS = zeros(nt,L);             % initialize pilot matrix for BS
    Y_UE = zeros(nr,L);             % initialize pilot matrix for UE
    H_her = transpose(H);           % getting transpose of H

    Pol_BS_t_blk_full = zeros(2*nt,nt,L);  % storage for BS blockdiag matrix
    Pol_BS_r_blk_full = zeros(2*nt,nt,L);  % storage for BS blockdiag matrix
    Pol_UE_t_blk_full = zeros(2*nr,nr,L);  % storage for UE blockdiag matrix
    Pol_UE_r_blk_full = zeros(2*nr,nr,L);  % storage for UE blockdiag matrix

    if mode == "LS_Method"
        for l = 1:L
            Pol_BS_t_blk_full(:,:,l) = make_blk_diag(Pol_BS_t);
            Pol_BS_r_blk_full(:,:,l) = make_blk_diag(Pol_BS_r);
            Pol_UE_t_blk_full(:,:,l) = make_blk_diag(Pol_UE_t);
            Pol_UE_r_blk_full(:,:,l) = make_blk_diag(Pol_UE_r);

            Pol_BS_t_blk = Pol_BS_t_blk_full(:,:,l);
            Pol_BS_r_blk = Pol_BS_r_blk_full(:,:,l);
            Pol_UE_t_blk = Pol_UE_t_blk_full(:,:,l);
            Pol_UE_r_blk = Pol_UE_r_blk_full(:,:,l);

            W = bf(:,:,l);
            % ---------UE observes the pilots-----------------
            y_UE_noiseless = Pol_UE_r_blk' * H * Pol_BS_t_blk * W;
            disp(y_UE_noiseless)
            % noise at UE side
            noise_real = random('Normal',0,1,nr,1);
            noise_imag = random('Normal',0,1,nr,1);
            noise = (noise_real + 1i*noise_imag) / sqrt(2);
            noise = noise * sqrt(N0);
            Y_UE(:,l) = y_UE_noiseless + noise;
    
            % ---------BS observes the pilots--------------
            y_BS_noiseless = Pol_BS_r_blk' * H_her * Pol_UE_t_blk;
            disp(y_BS_noiseless)
            % noise at BS side
            noise_real = random('Normal',0,1,nt,1);
            noise_imag = random('Normal',0,1,nt,1);
            noise = (noise_real + 1i*noise_imag) / sqrt(2);
            noise = noise * sqrt(N0);
            Y_BS(:,l) = y_BS_noiseless + noise;
        end
        
    else  
        for l = 1:L
            Pol_BS_t_blk_full(:,:,l) = make_blk_diag(Pol_BS_t(:,:,l));
            Pol_BS_r_blk_full(:,:,l) = make_blk_diag(Pol_BS_r(:,:,l));
            Pol_UE_t_blk_full(:,:,l) = make_blk_diag(Pol_UE_t(:,:,l));
            Pol_UE_r_blk_full(:,:,l) = make_blk_diag(Pol_UE_r(:,:,l));
           
            Pol_BS_t_blk = Pol_BS_t_blk_full(:,:,l);
            Pol_BS_r_blk = Pol_BS_r_blk_full(:,:,l);
            Pol_UE_t_blk = Pol_UE_t_blk_full(:,:,l);
            Pol_UE_r_blk = Pol_UE_r_blk_full(:,:,l);
           
            W = bf(:,:,l);
    
            % ---------UE observes the pilots-----------------
            y_UE_noiseless = Pol_UE_r_blk' * H * Pol_BS_t_blk * W;
    %         disp(y_UE_noiseless)
            % noise at UE side
            noise_real = random('Normal',0,1,nr,1);
            noise_imag = random('Normal',0,1,nr,1);
            noise = (noise_real + 1i*noise_imag) / sqrt(2);
            noise = noise * sqrt(N0);
            Y_UE(:,l) = y_UE_noiseless + noise;
    
            % ---------BS observes the pilots--------------
            y_BS_noiseless = Pol_BS_r_blk' * H_her * Pol_UE_t_blk;
    %         disp(y_BS_noiseless)
            % noise at BS side
            noise_real = random('Normal',0,1,nt,1);
            noise_imag = random('Normal',0,1,nt,1);
            noise = (noise_real + 1i*noise_imag) / sqrt(2);
            noise = noise * sqrt(N0);
            Y_BS(:,l) = y_BS_noiseless + noise;
        end
    end
    
end