function [Pol_BS, Pol_UE, Pol_BS_blk, Pol_UE_blk] = MC_Optimization(H_up, N, M, joint_iteration)
    angle_UE = rand(1,M) * pi;
    Pol_UE = [cos(angle_UE); sin(angle_UE)];
    Pol_UE_blk = make_blk_diag(Pol_UE);
    opt_theta_BS = zeros(1,N);
    opt_theta_UE = zeros(1,M);

    for k = 1:joint_iteration
        % BaseStation Polarization Optimization
        n = 1;
        H_PD_BS = H_up' * (Pol_UE_blk) * Pol_UE_blk' * H_up;
        for i = 1:2:2*N
            C = H_PD_BS(i:i+1,i:i+1);
            opt_theta_BS(n) = 0.5*atan(2*real(C(1,2))/(C(1,1)-C(2,2)));
            n = n+1;
        end
        opt_theta_BS = real(opt_theta_BS);
        Pol_BS = [cos(opt_theta_BS); sin(opt_theta_BS)];
        Pol_BS_blk = make_blk_diag(Pol_BS);

        % User Polarization Optimization
        m = 1;
        H_PD_UE = H_up * (Pol_BS_blk) * Pol_BS_blk' * H_up';
        for j = 1:2:2*M
            C = H_PD_UE(j:j+1,j:j+1);
            opt_theta_UE(m) = 0.5*atan(2*real(C(1,2))/(C(1,1)-C(2,2)));
            m = m+1;
        end
        opt_theta_UE = real(opt_theta_UE);
        Pol_UE = [cos(opt_theta_UE); sin(opt_theta_UE)];
        Pol_UE_blk = make_blk_diag(Pol_UE);
    end 
end