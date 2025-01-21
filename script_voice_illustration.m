%% Just try to get some feeling about two impulse responses% Implementation of the baseline, i.e., LS-estimation of the filter
% coefficients by only using L2 or L1 regularization on the parameters.

%% Generate some data
close all; clear all; clc
c = 343; % As in the simulation of rir
% fs = 8e3;
rng(0)
% rng(2)

D= 6 ;% Downsampling 
% Source trajectory: 
v_max = 1;
tdiff = 0.1; % Time between h estimates.
d_max = tdiff*v_max/2; % Maximum distance 

% Room properties
mic_pos = [0, 0, 0];
N_rir = 4;
reflection = 0.5;
room_dim = [4 6 3];
sigma2_t = 0;


% Randomize trajectory
% K = 15;
K = 10;
src_pos = zeros(3,K);
src_pos(:,1) = [2 3 1];
for k = 2:K
    dir = rand(3,1); % Direction to new source
    src_pos(:,k) = dir/norm(dir)*d_max + src_pos(:,k-1);
end 

% Just to get sampling frequency 
[x,fs] = audioread("m12_judge.wav");
x = x/abs(max(x));
fs = fs/D;

[h]=rir(fs, mic_pos, N_rir, reflection, room_dim, src_pos,sigma2_t);
% Nh = length(h)-500;
Nh = length(h)-400;

% --------- Load input data 
% [x,fs] = audioread("m12_judge.wav");
N_cuts = K;
N = 300+Nh*2;

start = 36e3;
stop = 36e3+N*N_cuts;
% stop = 54e3;
% N= stop-start;
x = x(start:stop);
% Let's divide signal into some smaller cuts
% N = floor(N/N_cuts);

xs = zeros(N,N_cuts);

for k = 1:K
    xs(:,k) = x((k-1)*N+1:(k-1)*N+N);
end


% --------- Generate the impulse responses
hs = zeros(Nh,K);
hs_gt = zeros(Nh,K);
for k = 1:K
    [h_true,h]=rir(fs, mic_pos, N_rir, reflection, room_dim, src_pos(:,k),sigma2_t);
    
    hs_gt(:,k) = h_true(1:Nh)/sum(abs(h_true(1:Nh)));
    hs(:,k) = h(1:Nh)/sum(abs(h(1:Nh))); % Normalize impulse response
end

% ------------ Generate output data
% N = round(Nh*1.5);
% xs = randn(N+Nh*2,K); % One of the Nh extra is for filter below, and one is for the convolution.
ys = zeros(N,N_cuts);
for k = 1:K
    ys(:,k) = filter(hs(:,k),1,xs(:,k))+  randn(size(xs,1),1)/100;
end

% Remove first Nh samples due to filtering artifacts
xs = xs(Nh+1:end,:);
ys = ys(Nh+1:end,:);
N = N-Nh;


% ------------ Prepare the convolution matrix
N = N-Nh; % Remove last Nh samples from x since they never are used to estimate something in y
Xs = zeros(N,Nh,K);
% Xs = zeros(N,Nh,K);

for k = 1:K
    for n = 1:N
        Xs(n,:,k) = flip(xs(n+1:n+Nh,k));
    end
end



%% Estimate the filter coefficients based on the signal 


cvx_begin 
    variable hs_est(Nh,K)

    J = 0;
    for k = 1:K
        J = J + norms(ys(Nh+1:end,k)-Xs(:,:,k)*hs_est(:,k));
    end

    minimize J

%     hs_est>=0;

cvx_end
hs_unreg = hs_est;

ploths(1:3,hs,hs_est)


%% Estimate with some tikhonov regularization 


% TODO: Cross-validation for choosing lambda
N_lambdas = 10;
lambdas = logspace(3,-3,N_lambdas);

hs_est_lambdas = zeros(Nh,K,N_lambdas);
NMSE_lambdas = zeros(N_lambdas,1);
for l = 1:N_lambdas
    
    cvx_begin quiet
        variable hs_est(Nh,K) 
    
        J = 0;
        for k = 1:K
            J = J + norms(ys(Nh+1:end,k)-Xs(:,:,k)*hs_est(:,k));
        end
    
        R = norms(hs_est(:));
        
        lambda = lambdas(l);
        minimize J+ lambda*R
    
    %     hs_est>=0;
    
    cvx_end

    hs_est_lambdas(:,:,l) = hs_est;
    NMSE_lambdas(l) = norms(hs_est(:)-hs(:))/norm(hs(:));
end

% Pick the one with smallest NMSE.
[~,I] = min(NMSE_lambdas);
hs_tik = hs_est_lambdas(:,:,I);


% Check so best lambda is within interval
fprintf("Lambda within interval: %d \n " , I>1 && I<N_lambdas)


ploths(1:3,hs,hs_tik)


%% And also with some sparsity


% TODO: Cross-validation for choosing lambda
N_lambdas = 10;
lambdas = logspace(3,-3,N_lambdas);

hs_est_lambdas = zeros(Nh,K,N_lambdas);
NMSE_lambdas = zeros(N_lambdas,1);
for l = 1:N_lambdas
    
    cvx_begin quiet
        variable hs_est(Nh,K)
    
        J = 0;
        for k = 1:K
            J = J + norms(ys(Nh+1:end,k)-Xs(:,:,k)*hs_est(:,k));
        end
    
        R = norms(hs_est(:),1);
        
        lambda = lambdas(l);
        minimize J+ lambda*R
    
    %     hs_est>=0;
    
    cvx_end

    hs_est_lambdas(:,:,l) = hs_est;
    NMSE_lambdas(l) = norms(hs_est(:)-hs(:))/norm(hs(:));
end

% Pick the one with smallest NMSE.
[~,I] = min(NMSE_lambdas);
hs_lasso = hs_est_lambdas(:,:,I);

ploths(1:3,hs,hs_lasso)

% Check so best lambda is within interval
fprintf("Lambda within interval: %d \n " , I>1 && I<N_lambdas)


%% Estimate with Tikhonov distance 

% TODO: Cross-validation for choosing lambda
N_lambdas = 10;
lambdas = logspace(3,-3,N_lambdas);

hs_est_lambdas = zeros(Nh,K,N_lambdas);
NMSE_lambdas = zeros(N_lambdas,1);
for l = 1:N_lambdas
    
    cvx_begin quiet
        variable hs_est(Nh,K) 
    
        J = 0;
        for k = 1:K
            J = J + norms(ys(Nh+1:end,k)-Xs(:,:,k)*hs_est(:,k));
        end
    
        R = 0;
        for k = 2:K
            R = R+ sum(square_abs(hs_est(:,k)-hs_est(:,k-1)));
        end

        lambda = lambdas(l);
        minimize J+ lambda*R
    
    %     hs_est>=0;
    
    cvx_end

    hs_est_lambdas(:,:,l) = hs_est;
    NMSE_lambdas(l) = norms(hs_est(:)-hs(:))/norm(hs(:));
end

% Pick the one with smallest NMSE.
[~,I] = min(NMSE_lambdas);
hs_tik_dist = hs_est_lambdas(:,:,I);


% Check so best lambda is within interval
fprintf("Lambda within interval: %d \n " , I>1 && I<N_lambdas)


ploths(1:3,hs,hs_tik_dist)


%% Estimate with L1 distance


% TODO: Cross-validation for choosing lambda
N_lambdas = 10;
lambdas = logspace(3,-3,N_lambdas);

hs_est_lambdas = zeros(Nh,K,N_lambdas);
NMSE_lambdas = zeros(N_lambdas,1);
for l = 1:N_lambdas
    
    cvx_begin quiet
        variable hs_est(Nh,K) 
    
        J = 0;
        for k = 1:K
            J = J + norms(ys(Nh+1:end,k)-Xs(:,:,k)*hs_est(:,k),2);
        end
    
        R = 0;
        for k = 2:K
            R = R+ sum(abs(hs_est(:,k)-hs_est(:,k-1)));
        end

        lambda = lambdas(l);
        minimize J+ lambda*R
    
    %     hs_est>=0;
    
    cvx_end

    hs_est_lambdas(:,:,l) = hs_est;
    NMSE_lambdas(l) = norms(hs_est(:)-hs(:))/norm(hs(:));
end

% Pick the one with smallest NMSE.
[~,I] = min(NMSE_lambdas);
hs_lasso_dist = hs_est_lambdas(:,:,I);


% Check so best lambda is within interval
fprintf("Lambda within interval: %d \n " , I>1 && I<N_lambdas)


ploths(1:3,hs,hs_tik_dist)

%% Do the same but with some OMT regularization 
% close all
% reweights = 1;
% 
% weights = ones(Nh,Nh,K);
% for r = 1:reweights
%     cvx_begin
%         variable hs_est(Nh,K)
%         variable Mass(Nh,Nh,K-1)
% 
%         % Data fitting term
%         J = 0;
%         for k = 1:K
%             J = J + norms(ys(Nh+1:end,k)-Xs(:,:,k)*hs_est(:,k));
%         end
%   
%         % Regularization term 
%         R = 0;
%         C = getOMTCostEuclidean(Nh,weights(:,:,k))/fs;
%         for k = 2:K
%             % Some temporarily variables
%             R = R+ sum(sum(C.*Mass(:,:,k-1)));        
%         end
% 
%         lambda = 1;
%         minimize J + lambda*R;
% 
%         subject to 
%             for k = 2:K
%                 sum(Mass(:,:,k-1),1) == hs_est(:,k)';
%                 sum(Mass(:,:,k-1),2) == hs_est(:,k-1);
%             end
%             Mass>=0;
%         
%     cvx_end
% end
% hs_omt = hs_est;
% 
% 
% ploths(1:3,hs,hs_omt)

%% Allow for positive energy




close all

% TODO: Cross-validation for choosing lambda
N_lambdas = 10;
lambdas = logspace(3,-3,N_lambdas);

hs_est_lambdas = zeros(Nh,K,N_lambdas);
NMSE_lambdas = zeros(N_lambdas,1);
for l = 1:N_lambdas
    
    
    cvx_begin 
        variable hs_est_pos(Nh,K)
        variable hs_est_neg(Nh,K)
        variable Mass_pos(Nh,Nh,K-1)
        variable Mass_neg(Nh,Nh,K-1)
    
        % Data fitting term
        J = 0;
        for k = 1:K
            J = J + norms(ys(Nh+1:end,k)-Xs(:,:,k)*(hs_est_pos(:,k)-hs_est_neg(:,k)));
    %          J = J + norms(ys(Nh+1:end,k)-Xs(:,:,k)*(hs_est_pos(:,k)));
        end
    
        % Regularization term 
        R = 0;
        C = getOMTCostEuclidean(Nh,ones(Nh,Nh))/fs + 1e-3;
        for k = 2:K
            R = R+ sum(sum(C.*Mass_pos(:,:,k-1)));    
            R = R+ sum(sum(C.*Mass_neg(:,:,k-1)));    
        end
    
        % To avoid both large negative and positive taps, we should try to
        % avoid them by 
%         R_extra = sum(Mass_pos(:)+Mass_neg(:));
        
        lambda = 10;
        lambda = lambdas(l);
        rho = 0.01;
        minimize J + lambda*(R)%+rho*R_extra);
    
        subject to 
            for k = 2:K
                sum(Mass_pos(:,:,k-1),1) == hs_est_pos(:,k)';
                sum(Mass_pos(:,:,k-1),2) == hs_est_pos(:,k-1);
    
                sum(Mass_neg(:,:,k-1),1) == hs_est_neg(:,k)';
                sum(Mass_neg(:,:,k-1),2) == hs_est_neg(:,k-1);
            end
            Mass_pos>=0;
            Mass_neg>=0;
    
    cvx_end
    hs_est = hs_est_pos-hs_est_neg;

    hs_est_lambdas(:,:,l) = hs_est;
    NMSE_lambdas(l) = norms(hs_est(:)-hs(:))/norm(hs(:));

end

% Pick the one with smallest NMSE.
[~,I] = min(NMSE_lambdas);
hs_omt = hs_est_lambdas(:,:,I);

ploths(1:3,hs,hs_omt)

% Check so best lambda is within interval
fprintf("Lambda within interval: %d \n " , I>1 && I<N_lambdas)


%% What's the NMSE of the impulse responses?

NMSE_omt = sum((hs(:)-hs_omt(:)).^2)/sum(hs(:).^2)
NMSE_lasso = sum((hs(:)-hs_lasso(:)).^2)/sum(hs(:).^2)
NMSE_lasso_dist = sum((hs(:)-hs_lasso_dist(:)).^2)/sum(hs(:).^2)
NMSE_tik = sum((hs(:)-hs_tik(:)).^2)/sum(hs(:).^2)
NMSE_tik_dist = sum((hs(:)-hs_tik_dist(:)).^2)/sum(hs(:).^2)
NMSE_unreg = sum((hs(:)-hs_unreg(:)).^2)/sum(hs(:).^2)


%% Heatmap of the impulse responses
% Max value 
h_max = max([max(hs(:)),max(hs_lasso(:)),max(hs_tik(:)), max(hs_omt(:))])/3;
h_min = 0;
close all;

fi = figure


subplot(161)
imagesc(hs)
ylabel("Tap, n")
xlabel("Impulse response, k")
xticks(1:K)
clim([h_min, h_max])
title("True",'Interpreter','latex')

subplot(162)
imagesc(hs_tik)
clim([h_min, h_max])
title("Tikhonov",'Interpreter','latex')
xticks(1:K)
xticklabels("")

subplot(163)
imagesc(hs_lasso)
clim([h_min, h_max])
title("Lasso",'Interpreter','latex')
xticks(1:K)
xticklabels("")

subplot(164)
imagesc(hs_tik_dist)
clim([h_min, h_max])
title("$d_{\ell 2}$",'Interpreter','latex')
xticks(1:K)
xticklabels("")

subplot(165)
imagesc(hs_lasso_dist)
clim([h_min, h_max])
title("$d_{\ell 1}$",'Interpreter','latex')
xticks(1:K)
xticklabels("")

subplot(166)
imagesc(hs_omt)
clim([h_min, h_max])
title("$d_{OT}$",'Interpreter','latex')
xticks(1:K)
xticklabels("")
% colormap(colormap(flipud(copper)))

% h = axes(fi,'visible','off'); 
% c = colorbar(h,'Position',[0.93 0.168 0.022 0.7]);  % attach colorbar to h



colormap((gray))

h = axes(fi,'visible','off'); 
c = colorbar(h,'Position',[0.93 0.168 0.022 0.7]);  % attach colorbar to h
caxis(h,[h_min,h_max])

%% Heatmap of the impulse responses - log scale
low_cut = 1e-3;
hs_log = log10(max(hs,low_cut));
hs_tik_log = log10(max(hs_tik,low_cut));
hs_lasso_log = log10(max(hs_lasso,low_cut));
hs_tik_dist_log = log10(max(hs_tik_dist,low_cut));
hs_lasso_dist_log = log10(max(hs_lasso_dist,low_cut));
hs_omt_log = log10(max(hs_omt,low_cut));



% Max value 
h_max = max([max(hs_log(:)),max(hs_lasso_log(:)),max(hs_tik_log(:)), max(hs_omt_log(:))]);
h_min = log10(low_cut);
close all;

fi = figure


subplot(161)
imagesc(hs_log)
ylabel("Tap, n")
xlabel("Impulse response, k")
xticks(1:K)
clim([h_min, h_max])
title("True",'Interpreter','latex')

subplot(162)
imagesc(hs_tik_log)
clim([h_min, h_max])
title("Tikhonov",'Interpreter','latex')
xticks(1:K)
xticklabels("")

subplot(163)
imagesc(hs_lasso_log)
clim([h_min, h_max])
title("Lasso",'Interpreter','latex')
xticks(1:K)
xticklabels("")

subplot(164)
imagesc(hs_tik_dist_log)
clim([h_min, h_max])
title("$d_{\ell 2}$",'Interpreter','latex')
xticks(1:K)
xticklabels("")

subplot(165)
imagesc(hs_lasso_dist_log)
clim([h_min, h_max])
title("$d_{\ell 1}$",'Interpreter','latex')
xticks(1:K)
xticklabels("")

subplot(166)
imagesc(hs_omt_log)
clim([h_min, h_max])
title("$\widehat{d_{OT}}$",'Interpreter','latex')
xticks(1:K)
xticklabels("")
% colormap(colormap(flipud(copper)))
colormap((gray))

h = axes(fi,'visible','off'); 
c = colorbar(h,'Position',[0.93 0.168 0.022 0.7]);  % attach colorbar to h
caxis(h,[h_min,h_max])

%% Plot by varying the frequence of the source
close all

% TODO: 
N_sig = 10000;
signal_full = randn(N_sig,1);

freqs = 100:100:1000;
N_freqs= length(freqs);

signal_true = zeros(N_sig,K);
signal_tik = zeros(N_sig,K);
signal_lasso = zeros(N_sig,K);
signal_omt = zeros(N_sig,K);
signal_tik_dist = zeros(N_sig,K);
signal_lasso_dist = zeros(N_sig,K);

NMSE_sig_tik = zeros(N_freqs,1);
NMSE_sig_lasso = zeros(N_freqs,1);
NMSE_sig_omt = zeros(N_freqs,1);
NMSE_sig_tik_dist = zeros(N_freqs,1);
NMSE_sig_lasso_dist = zeros(N_freqs,1);

for n = 2:N_freqs
    signal = bandpass(signal_full,[freqs(n-1),freqs(n)],fs);
    for k = 1:K
        signal_true(:,k) = filter(hs(:,k),1,signal);
        signal_tik(:,k) = filter(hs_tik(:,k),1,signal);
        signal_lasso(:,k) = filter(hs_lasso(:,k),1,signal);
        signal_omt(:,k) = filter(hs_omt(:,k),1,signal);
        signal_tik_dist(:,k) = filter(hs_tik_dist(:,k),1,signal);
        signal_lasso_dist(:,k) = filter(hs_lasso_dist(:,k),1,signal);
    end


    NMSE_sig_tik(n) = sum((signal_tik(:)-signal_true(:)).^2)/sum(signal_true(:).^2);
    NMSE_sig_lasso(n) = sum((signal_lasso(:)-signal_true(:)).^2)/sum(signal_true(:).^2); 
    NMSE_sig_omt(n) = sum((signal_omt(:)-signal_true(:)).^2)/sum(signal_true(:).^2);
    NMSE_sig_tik_dist(n) = sum((signal_tik_dist(:)-signal_true(:)).^2)/sum(signal_true(:).^2);
    NMSE_sig_lasso_dist(n) = sum((signal_lasso_dist(:)-signal_true(:)).^2)/sum(signal_true(:).^2);


end

figure
hold on 
plot(freqs(1:end-1),NMSE_sig_tik(2:end),'*-')
plot(freqs(1:end-1),NMSE_sig_lasso(2:end),'.-')
plot(freqs(1:end-1),NMSE_sig_tik_dist(2:end),'o-')
plot(freqs(1:end-1),NMSE_sig_lasso_dist(2:end),'^-')
plot(freqs(1:end-1),NMSE_sig_omt(2:end),'x-')
hold off
grid on
hl = legend("Tikhonov","Lasso","$d_{\ell 2}$","$d_{\ell 1}$","$\widehat{d_{OT}}$");
set(hl, 'Interpreter','latex')

xticklabels({freqs(1) + "-"+ freqs(2),freqs(2) + "-"+ freqs(3),freqs(3) + "-"+ freqs(4),freqs(4) + "-"+ freqs(5),freqs(5) + "-"+ freqs(6), ...
    freqs(6) + "-"+ freqs(7),freqs(7) + "-"+ freqs(8),freqs(8) + "-"+ freqs(9),freqs(9) + "-"+ freqs(10)})
xlabel("Frequency, hz")
ylabel("NMSE_{signal}")

%% Plot of variations in number of points

