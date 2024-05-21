function [Pol_BS_t,Pol_BS_r,Pol_UE_t,Pol_UE_r,beamformer] = get_pilot_parameters(L,nt,nr,Pt)
    % BS polarization
    % transmit
    rand_angle_t = rand(1,nt) * pi;
    Pol_BS_t = [cos(rand_angle_t); sin(rand_angle_t)];
    % receive
    rand_angle_r = rand(1,nt,4) * pi;
    Pol_BS_r = [cos(rand_angle_r); sin(rand_angle_r)];

    % UE polarization
    % transmit
    rand_angle_t = rand(1,nr) * pi;
    Pol_UE_t = [cos(rand_angle_t); sin(rand_angle_t)];
    % receive
    rand_angle_r = rand(1,nr) * pi;
    Pol_UE_r = [cos(rand_angle_r); sin(rand_angle_r)];
    
    % beamformer 
    W_real = random('Normal',0,1,nt,1,L);
    W_imag = random('Normal',0,1,nt,1,L);
    W = W_real + 1i*W_imag;
    W_norm = sqrt(sum(abs(W).^2));
    beamformer = W./W_norm * sqrt(Pt);
end