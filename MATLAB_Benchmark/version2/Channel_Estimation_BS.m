function H_BS_hat = Channel_Estimation_BS(Y,A,B)
    C = A'*Y*B';
    F = A'*A;
    G = B*B';
    H_BS_hat = pinv(F)*C*pinv(G);
end