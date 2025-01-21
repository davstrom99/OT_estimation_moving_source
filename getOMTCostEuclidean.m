function [C] = getOMTCost(N,W)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    C = zeros(N,N);
    for i = 1:N
        for j = 1:N
            C(i,j) = abs(i-j)^2*W(i,j);
        end
    end




end