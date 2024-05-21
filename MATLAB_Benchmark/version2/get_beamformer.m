function W = get_beamformer(Heff, Pt, mode)
    [~,nt]=size(Heff);
    if (mode == "random")
        W_real = random('Normal',0,1,nt,1);
        W_imag = random('Normal',0,1,nt,1);
        W = W_real + 1i*W_imag;
        W_norm = sqrt(sum(abs(W).^2));
        W = W/W_norm * sqrt(Pt);
    else 
        W = Heff';
        W_norm = sqrt(sum(abs(W).^2));
        W = W/W_norm * sqrt(Pt);
    end
end