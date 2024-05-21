function [Pol_BS_t,Pol_BS_r,Pol_UE_t,Pol_UE_r,beamformer] = get_pilot_parameters(L,nt,nr,Pt,mode)
    if mode == "random"
        % BS polarization
        rand_angle_tx = rand(1,nt,L) * pi;
        Pol_BS_t = [cos(rand_angle_tx); sin(rand_angle_tx)];
        Pol_BS_r = Pol_BS_t;

        % UE polarization
        rand_angle_rx = rand(1,nr,L) * pi;
        Pol_UE_t = [cos(rand_angle_rx); sin(rand_angle_rx)];
        Pol_UE_r = Pol_UE_t;
    elseif mode == "LS_Method"
        % BS polarization
        % transmit
        rand_angle_t = rand(1,nt) * pi;
        Pol_BS_t = [cos(rand_angle_t); sin(rand_angle_t)];
        % receive
        rand_angle_r = rand(1,nt) * pi;
        Pol_BS_r = [cos(rand_angle_r); sin(rand_angle_r)];

        % UE polarization
        % transmit
        rand_angle_t = rand(1,nr) * pi;
        Pol_UE_t = [cos(rand_angle_t); sin(rand_angle_t)];
        % receive
        rand_angle_r = rand(1,nr) * pi;
        Pol_UE_r = [cos(rand_angle_r); sin(rand_angle_r)];
    elseif mode == "tpi4_rVH"
        % BS polarization
        % transmit
        rand_angle =  rand * pi;
        angle_BS_t = ones(1,nt,L);
        angle_BS_t = angle_BS_t * rand_angle;
        Pol_BS_t = [cos(angle_BS_t); sin(angle_BS_t)];
        % receive
        angle_BS_r = zeros(1,nt,L);
        angle_BS_r(:,:,2:2:end)=1; 
        angle_BS_r = angle_BS_r * (pi/2);
        Pol_BS_r = [cos(angle_BS_r); sin(angle_BS_r)];

        % UE polarization
        % transmit
        rand_angle = rand * pi;
        angle_UE_t = ones(1,nr,L);
        angle_UE_t = angle_UE_t * rand_angle;
        Pol_UE_t = [cos(angle_UE_t); sin(angle_UE_t)];
        % receive
        angle_UE_r = zeros(1,nr,L); angle_UE_r(:,:,2:2:end)=1;
        angle_UE_r = angle_UE_r * (pi/2);
        Pol_UE_r = [cos(angle_UE_r); sin(angle_UE_r)];
    else 
        % BS polarization
        % transmit
        angle_BS_t = zeros(1,nt,L);
        angle_BS_t(:,:,2:2:end)=1;
        angle_BS_t = angle_BS_t * (pi/2);
        Pol_BS_t = [cos(angle_BS_t); sin(angle_BS_t)];
        % receive
        angle_BS_r = zeros(1,nt,L);
        angle_BS_r(:,:,3:5:end)=1; angle_BS_r(:,:,4:5:end)=1;
        angle_BS_r = angle_BS_r * (pi/2);
        Pol_BS_r = [cos(angle_BS_r); sin(angle_BS_r)];

        % UE polarization
        % transmit
        angle_UE_t = zeros(1,nr,L); angle_UE_t(:,:,2:2:end)=1;
        angle_UE_t = angle_UE_t * (pi/2);
        Pol_UE_t = [cos(angle_UE_t); sin(angle_UE_t)];
        % receive
        angle_UE_r = zeros(1,nr,L);
        angle_UE_r(:,:,3:5:end)=1; angle_UE_r(:,:,4:5:end)=1;
        angle_UE_r = angle_UE_r * (pi/2);
        Pol_UE_r = [cos(angle_UE_r); sin(angle_UE_r)];
    end
    % beamformer 
    W_real = random('Normal',0,1,nt,1,L);
    W_imag = random('Normal',0,1,nt,1,L);
    W = W_real + 1i*W_imag;
    W_norm = sqrt(sum(abs(W).^2));
    beamformer = W./W_norm * sqrt(Pt);
end