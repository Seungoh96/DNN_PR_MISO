function C = calculate_capacity(Heff, W, N0)
    gain = abs(Heff*W)^2;
    C = log2(1 + gain/N0);
end