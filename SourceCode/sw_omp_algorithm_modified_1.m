function [estimated_Hk,x_upt_temp,psiMat_cols,Gamma_w_new,AoD_angles_arr,AoA_angles_arr,Gt,Gr,At_1,Ar_1,At_cols,Ar_cols] = sw_omp_algorithm_modified_1(Nfft,Ntrain,var_n,D_w,PhiMat,A,At_1,Ar_1,r,psiMat,Lr,Gt,Gr,Nt,Nr)

Gamma_w = D_w'\A;  % sw-omp
Gamma_w_new = Gamma_w;
psiMat_1 = psiMat;
AoD_angles_arr = 1:Gt;
AoA_angles_arr = 1:Gr;
%%%%%%%%%%sw-omp%%%%%%%%%%%%%%%%%%%%%%%%%%
y_w = D_w'\r;
y_w1 = y_w;
iter = 0;

count_swomp_max = 5;
Gt_orig = Gt;
Gr_orig = Gr;
DictSize = 128;
Gamma_w_support = [];
psiMat_cols = [];
At_cols = [];
Ar_cols = [];
while(1)
    iter = iter + 1;
    Gt = Gt_orig;
    Gr = Gr_orig;
    
    c = Gamma_w'*y_w1;
    
    count_swomp = 0;
    maxVal_prev = 0;
    while 1
        
        count_swomp = count_swomp+1;
        
        c = abs(c);
        [maxVal,Index] = max(sum(c,2));
        
        if abs(maxVal_prev-maxVal)/maxVal_prev<1e-10 || count_swomp==count_swomp_max
            break;
        end
        maxVal_prev = maxVal;
        
        
        if count_swomp == 1
            AoD_angles = ceil(Index/Gr);
            AoA_angles = Index-(AoD_angles-1)*Gr;
            
            if AoD_angles>5
                theta_AoD_min = acos(1-2/Gt*(AoD_angles-5-1));
            else
                theta_AoD_min = 0;
            end
            if AoD_angles<Gt-5
                theta_AoD_max = acos(1-2/Gt*(AoD_angles+5));
            else
                theta_AoD_max = pi;
            end
            if AoA_angles>5
                theta_AoA_min = acos(1-2/Gr*(AoA_angles-5-1));
            else
                theta_AoA_min = 0;
            end
            if AoA_angles<Gr-5
                theta_AoA_max = acos(1-2/Gr*(AoA_angles+5));
            else
                theta_AoA_max = pi;
            end
            
        else
            AoD_angles = ceil(Index/DictSize);
            AoA_angles = Index-(AoD_angles-1)*DictSize;
            if AoD_angles>DictSize/4
                theta_AoD_min = (theta_AoD_range(AoD_angles-floor(DictSize/4)));%+theta_AoD_range(AoD_angles))/2;
            end
            if AoD_angles<3*DictSize/4
                theta_AoD_max = theta_AoD_range(AoD_angles+floor(DictSize/4));
            end
            if AoA_angles>DictSize/4
                theta_AoA_min = (theta_AoA_range(AoA_angles-floor(DictSize/4)));%+theta_AoA_range(AoA_angles))/2;
            end
            if AoA_angles<3*DictSize/4
                theta_AoA_max = theta_AoA_range(AoA_angles+floor(DictSize/4));
            end
        end
        
        theta_AoD_step = (theta_AoD_max-theta_AoD_min)/(DictSize-1);
        theta_AoA_step = (theta_AoA_max-theta_AoA_min)/(DictSize-1);
        theta_AoD_range = theta_AoD_min:theta_AoD_step:theta_AoD_max;
        theta_AoA_range = theta_AoA_min:theta_AoA_step:theta_AoA_max;
        At_1 = 1/sqrt(Nt) * exp(pi*1j * [0:Nt-1]' * cos(theta_AoD_range));
        Ar_1 = 1/sqrt(Nr) * exp(pi*1j * [0:Nr-1]' * cos(theta_AoA_range));
        
        psiMat_1 = kron(conj(At_1),Ar_1);
        
        Gamma_w_new = D_w'\(PhiMat*psiMat_1);  % sw-omp
        c = Gamma_w_new'*y_w1;
    end
    AoD_angles = ceil(Index/DictSize);
    AoA_angles = Index-(AoD_angles-1)*DictSize;
    At_cols = [At_cols 1/sqrt(Nt)*exp(pi*1j*[0:Nt-1]'*cos(theta_AoD_range(AoD_angles)))];
    Ar_cols = [Ar_cols 1/sqrt(Nr)*exp(pi*1j*[0:Nr-1]'*cos(theta_AoA_range(AoA_angles)))];
    
    Gamma_w_support = [Gamma_w_support Gamma_w_new(:,Index)];
    psiMat_cols = [psiMat_cols psiMat_1(:,Index)];
    x_upt_temp = (Gamma_w_support'*Gamma_w_support+0*var_n*eye(size(Gamma_w_support,2)))\Gamma_w_support' * y_w;
    y_w1 = y_w - Gamma_w_support*x_upt_temp;
    MSE = norm(y_w1,'fro')^2;
    
    MSE = MSE/(Nfft*Ntrain*Lr);
    %     if((MSE<epsilon && iter>=5) || iter == 10)
    if(iter == 16)
        break;
    end
end

estimated_Hk = psiMat_cols*x_upt_temp;
end

