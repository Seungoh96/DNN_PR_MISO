function H_hat = LS_Estimation(Y,X)
    H_hat = Y*X'*pinv(X*X'); 
end