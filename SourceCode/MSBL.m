function [X_mean_est, N_Iter, gamma_vec] = MSBL(PhiMat, y, NoiseVar, CONVERGENCE_TOL, MAX_ITER)

M = size(PhiMat, 1);
N = size(PhiMat, 2);
L = size(y, 2);

%% Initialization of hyperparameters
gamma_vec = ones(N, 1);

%% Computation of the inverse covariance matrix of the sparse vector
gamma_diagMat = diag(gamma_vec);
PhiGammaHerm = gamma_diagMat*PhiMat';
sigma_ymat = NoiseVar*eye(M) + PhiMat*gamma_diagMat*PhiMat';
sigma_x_mat = gamma_diagMat-PhiGammaHerm/sigma_ymat*PhiGammaHerm';

X_mean_est = zeros(N,L);

MatchedFilt_Out = PhiMat' * y;
N_Iter = 0;
TotalTime = 0;
while(1)
    
    N_Iter = N_Iter + 1;
    
    X_mean_est_prev = X_mean_est;
    
    %% Computation of the covariance matrix of the sparse vector
    gamma_vec = real(diag(1/L * (X_mean_est * X_mean_est') + sigma_x_mat));
    gamma_diagMat = diag(gamma_vec);
    
    PhiGammaHerm = gamma_diagMat*PhiMat';
    sigma_ymat = NoiseVar*eye(M) + PhiMat*gamma_diagMat*PhiMat';
    sigma_x_mat = gamma_diagMat - PhiGammaHerm/sigma_ymat*PhiGammaHerm';
    
    %% Computation of the mean of the sparse vector estimate
    X_mean_est = (1/NoiseVar) * sigma_x_mat * MatchedFilt_Out;
    
    %% Stopping criterion
    X_diff = X_mean_est - X_mean_est_prev;
    diffnorm = norm(X_diff,'fro') / norm(X_mean_est_prev,'fro');
    
    timePerIter = toc;
    TotalTime = TotalTime + timePerIter;
    
    if(diffnorm < CONVERGENCE_TOL|| N_Iter>=MAX_ITER)
        break;
    end
    
end
end
