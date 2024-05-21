function H_UE_hat = Channel_Estimation_UE(Y,A,B)
    F = A'*A;
    G = B*B';
    C = A'*Y*B';
    H_UE_hat = pinv(F)*C*pinv(G);
end