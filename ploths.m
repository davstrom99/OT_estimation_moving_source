function [outputArg1,outputArg2] = ploths(Ks,hs,hs2,hs3,hs4)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here



%     K = size(hs,2);
    K = length(Ks);

    f = figure;
    f.WindowState = 'maximized';
    for k = 1:K
        subplot(K,1,k)
        plot(hs(:,Ks(k)))
        hold on 
        if nargin>2
        plot(hs2(:,Ks(k)),'--')
        end
        if nargin>3
        plot(hs3(:,Ks(k)),'.--')
        end
        if nargin>4
        plot(hs4(:,Ks(k)),'.-')
        end

        hold off
    
    end
    legend("True","Estimated")

end