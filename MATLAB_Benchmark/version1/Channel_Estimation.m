function [H_BS_hat_combined, H_UE_hat_combined] = Channel_Estimation(Y_BS, Y_UE, X_BS, X_UE, mode)
    nt = size(Y_BS,1);      % number of transmitter

    if mode == "LS"
        % BS side estimation protocol
        H_BS_hat = LS_Estimation(Y_BS, X_BS);
        H_BS_hat_combined = zeros(nt,4);
        counter = 0;
        for i = 1:4:size(H_BS_hat,2)
            H_BS_hat_combined = H_BS_hat_combined + H_BS_hat(:,i:i+4-1);
            counter = counter + 1;
        end
        H_BS_hat_combined = H_BS_hat_combined / counter;
        H_BS_hat_combined = reshape(transpose(reshape(transpose(H_BS_hat_combined),[],1)),2,[]);
       
        % UE side estimation protocol
        counter = 0;
        index = 1;
        H_UE_hat_combined_upper = 0;
        H_UE_hat_combined_bottom = 0;
        H_UE_hat = LS_Estimation(Y_UE, X_UE);
        for i = 1:2*nt:length(H_UE_hat)
            if mod(index,2) == 1
                H_UE_hat_combined_upper = H_UE_hat_combined_upper + H_UE_hat(i:i+2*nt-1);
            else
                H_UE_hat_combined_bottom = H_UE_hat_combined_bottom + H_UE_hat(i:i+2*nt-1);
            end   
            counter = counter + 1;
            index = index + 1;
        end
        H_UE_hat_combined_upper = H_UE_hat_combined_upper / (counter/2);
        H_UE_hat_combined_bottom = H_UE_hat_combined_bottom / (counter/2);
        H_UE_hat_combined = [H_UE_hat_combined_upper; H_UE_hat_combined_bottom];

    end
    
end