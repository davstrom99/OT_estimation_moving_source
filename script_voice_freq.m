%% Just try to get some feeling about two impulse responses% Implementation of the baseline, i.e., LS-estimation of the filter
% coefficients by only using L2 or L1 regularization on the parameters.

%% Generate some data
close all; clear all; clc
c = 343; % As in the simulation of rir
% fs = 8e3;
rng(0)
% rng(2)

Nmc = 10; % Number of monte carlos

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
for i_mc = 1:Nmc
    src_pos(:,1,i_mc) = [2 3 1];
    for k = 2:K
        dir = rand(3,1); % Direction to new source
        src_pos(:,k,i_mc) = dir/norm(dir)*d_max + src_pos(:,k-1,i_mc);
    end 
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
N = 300+Nh*2;

start = 36e3;
stop = 36e3+N*K;
% stop = 54e3;
% N= stop-start;
x = x(start:stop);
% Let's divide signal into some smaller cuts
% N = floor(N/N_cuts);

xs = zeros(N,K);


for k = 1:K
    xs(:,k) = x((k-1)*N+1:(k-1)*N+N);
end


% --------- Generate the impulse responses
hs = zeros(Nh,K,Nmc);
hs_gt = zeros(Nh,K,Nmc);
for i_mc = 1:Nmc
    for k = 1:K
        [h_true,h]=rir(fs, mic_pos, N_rir, reflection, room_dim, src_pos(:,k,i_mc),sigma2_t);
        
        hs_gt(:,k,i_mc) = h_true(1:Nh)/sum(abs(h_true(1:Nh)));
        hs(:,k,i_mc) = h(1:Nh)/sum(abs(h(1:Nh))); % Normalize impulse response
    end
end
% ------------ Generate output data
% N = round(Nh*1.5);
% xs = randn(N+Nh*2,K); % One of the Nh extra is for filter below, and one is for the convolution.
ys = zeros(N,K,Nmc);
for i_mc = 1:Nmc
    for k = 1:K
        ys(:,k,i_mc) = filter(hs(:,k,i_mc),1,xs(:,k))+  randn(size(xs,1),1)/100;
    end
end

% Remove first Nh samples due to filtering artifacts
xs = xs(Nh+1:end,:);
ys = ys(Nh+1:end,:,:);
N = N-Nh;


% ------------ Prepare the convolution matrix
N = N-Nh; % Remove last Nh samples from x since they never are used to estimate something in y
Xs = zeros(N,Nh,K,Nmc);
% Xs = zeros(N,Nh,K);
for i_mc = 1:Nmc
    for k = 1:K
        for n = 1:N
            Xs(n,:,k,i_mc) = flip(xs(n+1:n+Nh,k));
        end
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
% Set up the cross validations
N_lambdas = 10;
lambdas = logspace(3,-3,N_lambdas);

hs_tik = zeros(Nh,K,Nmc);
"Running cross validation for Tikhonov"
hs_est_lambdas = zeros(Nh,K,N_lambdas);
NMSE_lambdas = zeros(N_lambdas,1);
for l = 1:N_lambdas
    cvx_begin quiet
        variable hs_est(Nh,K) 
    
        J = 0;
        for k = 1:K
            J = J + norms(ys(Nh+1:end,k,1)-Xs(:,:,k,1)*hs_est(:,k));
        end
    
        R = norms(hs_est(:));
        
        lambda = lambdas(l);
        minimize J+ lambda*R
        
    cvx_end

    hs_est_lambdas(:,:,l) = hs_est;
    hs_first = hs(:,:,1);
    NMSE_lambdas(l) = norms(hs_est(:)-hs_first(:))/norm(hs_first(:));
end

% Pick the one with smallest NMSE.
[~,I] = min(NMSE_lambdas);
hs_tik(:,:,1) = hs_est_lambdas(:,:,I);
lambda = lambdas(I);
% Check so best lambda is within interval
fprintf("Lambda within interval: %d \n " , I>1 && I<N_lambdas)

% Then run for rest of the simulations whith this choice of lambda
for i_mc = 1:Nmc
    cvx_begin quiet
        variable hs_est(Nh,K) 
        J = 0;
        for k = 1:K
            J = J + norms(ys(Nh+1:end,k,i_mc)-Xs(:,:,k,i_mc)*hs_est(:,k));
        end
        R = norms(hs_est(:));
        minimize J+ lambda*R
    cvx_end

    hs_tik(:,:,i_mc) = hs_est;
end

% ploths(1:3,hs,hs_tik)

%% And also with some sparsity
% lambdas
N_lambdas = 10;
lambdas = logspace(3,-3,N_lambdas);

hs_lasso = zeros(Nh,K,Nmc);
"Running cross validation for Lasso"
hs_est_lambdas = zeros(Nh,K,N_lambdas);
NMSE_lambdas = zeros(N_lambdas,1);
for l = 1:N_lambdas
    
    cvx_begin quiet
        variable hs_est(Nh,K)
    
        J = 0;
        for k = 1:K
            J = J + norms(ys(Nh+1:end,k,1)-Xs(:,:,k,1)*hs_est(:,k));
        end
    
        R = norms(hs_est(:),1);
        
        lambda = lambdas(l);
        minimize J+ lambda*R
       
    cvx_end

    hs_est_lambdas(:,:,l) = hs_est;
    hs_first = hs(:,:,1);
    NMSE_lambdas(l) = norms(hs_est(:)-hs_first(:))/norm(hs_first(:));
end

% Pick the one with smallest NMSE.
[~,I] = min(NMSE_lambdas);
hs_lasso(:,:,1) = hs_est_lambdas(:,:,I);
lambda = lambdas(I);
% Check so best lambda is within interval
fprintf("Lambda within interval: %d \n " , I>1 && I<N_lambdas)

% Then run for rest of the simulations whith this choice of lambda
for i_mc = 1:Nmc
    cvx_begin quiet
        variable hs_est(Nh,K) 
        J = 0;
        for k = 1:K
            J = J + norms(ys(Nh+1:end,k,i_mc)-Xs(:,:,k,i_mc)*hs_est(:,k));
        end
        R = norms(hs_est(:),1);
        minimize J+ lambda*R
    cvx_end

    hs_lasso(:,:,i_mc) = hs_est;
end

% ploths(1:3,hs,hs_lasso)


%% Estimate with Tikhonov distance 

% lambdas
N_lambdas = 10;
lambdas = logspace(3,-3,N_lambdas);

hs_tik_dist = zeros(Nh,K,Nmc);
"Running cross validation for L2 distance"
hs_est_lambdas = zeros(Nh,K,N_lambdas);
NMSE_lambdas = zeros(N_lambdas,1);
for l = 1:N_lambdas
    cvx_begin quiet
        variable hs_est(Nh,K) 
    
        J = 0;
        for k = 1:K
            J = J + norms(ys(Nh+1:end,k,1)-Xs(:,:,k,1)*hs_est(:,k));
        end
    
        R = 0;
        for k = 2:K
            R = R+ sum(square_abs(hs_est(:,k)-hs_est(:,k-1)));
        end

        lambda = lambdas(l);
        minimize J+ lambda*R
   
    cvx_end

    hs_est_lambdas(:,:,l) = hs_est;
    hs_first = hs(:,:,1);
    NMSE_lambdas(l) = norms(hs_est(:)-hs_first(:))/norm(hs_first(:));
end

% Pick the one with smallest NMSE.
[~,I] = min(NMSE_lambdas);
hs_tik_dist(:,:,1) = hs_est_lambdas(:,:,I);
lambda = lambdas(I);
% Check so best lambda is within interval
fprintf("Lambda within interval: %d \n " , I>1 && I<N_lambdas)

for i_mc = 1:Nmc
    cvx_begin quiet
        variable hs_est(Nh,K) 
    
        J = 0;
        for k = 1:K
            J = J + norms(ys(Nh+1:end,k,i_mc)-Xs(:,:,k,i_mc)*hs_est(:,k));
        end
    
        R = 0;
        for k = 2:K
            R = R+ sum(square_abs(hs_est(:,k)-hs_est(:,k-1)));
        end

        minimize J+ lambda*R
   
    cvx_end

    hs_tik_dist(:,:,i_mc) = hs_est;
end




ploths(1:3,hs(:,:,1),hs_tik_dist(:,:,1))


%% Estimate with L1 distance
% Lambda
N_lambdas = 10;
lambdas = logspace(3,-3,N_lambdas);


hs_lasso_dist = zeros(Nh,K,Nmc);

"Running cross validation for L1 distance"
hs_est_lambdas = zeros(Nh,K,N_lambdas);
NMSE_lambdas = zeros(N_lambdas,1);
for l = 1:N_lambdas
    
    cvx_begin quiet
        variable hs_est(Nh,K) 
    
        J = 0;
        for k = 1:K
            J = J + norms(ys(Nh+1:end,k,1)-Xs(:,:,k,1)*hs_est(:,k),2);
        end
    
        R = 0;
        for k = 2:K
            R = R+ sum(abs(hs_est(:,k)-hs_est(:,k-1)));
        end

        lambda = lambdas(l);
        minimize J+ lambda*R
    cvx_end

    hs_est_lambdas(:,:,l) = hs_est;
    hs_first = hs(:,:,1);
    NMSE_lambdas(l) = norms(hs_est(:)-hs_first(:))/norm(hs_first(:));
end

% Pick the one with smallest NMSE.
[~,I] = min(NMSE_lambdas);
hs_lasso_dist(:,:,1) = hs_est_lambdas(:,:,I);
lambda = lambdas(I);
% Check so best lambda is within interval
fprintf("Lambda within interval: %d \n " , I>1 && I<N_lambdas)

for i_mc = 1:Nmc
    cvx_begin quiet
        variable hs_est(Nh,K) 
    
        J = 0;
        for k = 1:K
            J = J + norms(ys(Nh+1:end,k,i_mc)-Xs(:,:,k,i_mc)*hs_est(:,k));
        end
    
        R = 0;
        for k = 2:K
            R = R+ sum(abs(hs_est(:,k)-hs_est(:,k-1)));
        end

        minimize J+ lambda*R
   
    cvx_end

    hs_lasso_dist(:,:,i_mc) = hs_est;
end


ploths(1:3,hs(:,:,1),hs_lasso_dist(:,:,1))

%% Allow for positive energy
close all
% lambdas
N_lambdas = 10;
lambdas = logspace(3,-3,N_lambdas);

hs_omt = zeros(Nh,K,Nmc);

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
            J = J + norms(ys(Nh+1:end,k,1)-Xs(:,:,k,1)*(hs_est_pos(:,k)-hs_est_neg(:,k)));
    %          J = J + norms(ys(Nh+1:end,k)-Xs(:,:,k)*(hs_est_pos(:,k)));
        end
    
        % Regularization term 
        R = 0;
        C = getOMTCostEuclidean(Nh,ones(Nh,Nh))/fs + 1e-3;
        for k = 2:K
            R = R+ sum(sum(C.*Mass_pos(:,:,k-1)));    
            R = R+ sum(sum(C.*Mass_neg(:,:,k-1)));    
        end
    
       
        lambda = lambdas(l);
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
    hs_first = hs(:,:,1);
    NMSE_lambdas(l) = norms(hs_est(:)-hs_first(:))/norm(hs_first(:));
end

% Pick the one with smallest NMSE.
[~,I] = min(NMSE_lambdas);
hs_omt(:,:,1) = hs_est_lambdas(:,:,I);
lambda = lambdas(I);
% Check so best lambda is within interval
fprintf("Lambda within interval: %d \n " , I>1 && I<N_lambdas)



for i_mc = 1:Nmc
    cvx_begin 
        variable hs_est_pos(Nh,K)
        variable hs_est_neg(Nh,K)
        variable Mass_pos(Nh,Nh,K-1)
        variable Mass_neg(Nh,Nh,K-1)
    
        % Data fitting term
        J = 0;
        for k = 1:K
            J = J + norms(ys(Nh+1:end,k,i_mc)-Xs(:,:,k,i_mc)*(hs_est_pos(:,k)-hs_est_neg(:,k)));
    %          J = J + norms(ys(Nh+1:end,k)-Xs(:,:,k)*(hs_est_pos(:,k)));
        end
    
        % Regularization term 
        R = 0;
        C = getOMTCostEuclidean(Nh,ones(Nh,Nh))/fs + 1e-3;
        for k = 2:K
            R = R+ sum(sum(C.*Mass_pos(:,:,k-1)));    
            R = R+ sum(sum(C.*Mass_neg(:,:,k-1)));    
        end
    
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

    hs_omt(:,:,i_mc) = hs_est;
end



% ploths(1:3,hs,hs_omt)


%% What's the NMSE of the impulse responses?

NMSE_omt = sum((hs(:)-hs_omt(:)).^2)/sum(hs(:).^2)
NMSE_lasso = sum((hs(:)-hs_lasso(:)).^2)/sum(hs(:).^2)
NMSE_lasso_dist = sum((hs(:)-hs_lasso_dist(:)).^2)/sum(hs(:).^2)
NMSE_tik = sum((hs(:)-hs_tik(:)).^2)/sum(hs(:).^2)
NMSE_tik_dist = sum((hs(:)-hs_tik_dist(:)).^2)/sum(hs(:).^2)
% NMSE_unreg = sum((hs(:)-hs_unreg(:)).^2)/sum(hs(:).^2)


%% Plot by varying the frequence of the source
close all

% TODO: 
N_sig = 10000;
signal_full = randn(N_sig,1);

freqs = 100:100:1000;
N_freqs= length(freqs);

signal_true = zeros(N_sig,K,Nmc);
signal_tik = zeros(N_sig,K,Nmc);
signal_lasso = zeros(N_sig,K,Nmc);
signal_omt = zeros(N_sig,K,Nmc);
signal_tik_dist = zeros(N_sig,K,Nmc);
signal_lasso_dist = zeros(N_sig,K,Nmc);

NMSE_sig_tik = zeros(N_freqs,1);
NMSE_sig_lasso = zeros(N_freqs,1);
NMSE_sig_omt = zeros(N_freqs,1);
NMSE_sig_tik_dist = zeros(N_freqs,1);
NMSE_sig_lasso_dist = zeros(N_freqs,1);

for n = 2:N_freqs
    signal = bandpass(signal_full,[freqs(n-1),freqs(n)],fs);
    for i_mc = 1:Nmc
        for k = 1:K
            signal_true(:,k,i_mc) = filter(hs(:,k,i_mc),1,signal);
            signal_tik(:,k,i_mc) = filter(hs_tik(:,k,i_mc),1,signal);
            signal_lasso(:,k,i_mc) = filter(hs_lasso(:,k,i_mc),1,signal);
            signal_omt(:,k,i_mc) = filter(hs_omt(:,k,i_mc),1,signal);
            signal_tik_dist(:,k,i_mc) = filter(hs_tik_dist(:,k,i_mc),1,signal);
            signal_lasso_dist(:,k,i_mc) = filter(hs_lasso_dist(:,k,i_mc),1,signal);
        end
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


