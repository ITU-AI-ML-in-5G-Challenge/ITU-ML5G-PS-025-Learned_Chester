clc;
clear;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initilization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%data_set = 1; % choose the test data set between 1-9
%System parameters
Nt = 16; % Number of TX antennas
Nr = 64; % Number of RX antennas
Nbits= 4; % Number of bits available to represent a phase shift in the analog precoder/combiner.
Lt = 2;  % Number of TX RF chains
Lr = 4;  % Number of RX RF chains
Ns = 2;  % Number of data streams to be transmitted
Nfft = 256; % Number of subcarriers in the MIMO-OFDM system
Pt = 1; % Transmit power(mw)
%Nfilter = 20;
Mfilter = 1; %no oversampling
rolloff = 0.8;
MHz = 1e6;
fs = 1760*MHz; %Sampling frequency
Ts = 1/fs;
Nres = 2^Nbits; %resolution ofr the phase shifters

Run_SWOMP = 1;
MSBL_CONVERGENCE_TOL = 1e-6;
MSBL_MAX_ITER = 1000;

for data_set = 1:9
    %% Load datasets
    switch data_set
        case 1
            chan_save_file = 'test_dataset_v3_20_pilots_1_data_set.hdf5';
            load('prec_comb_sig_20_pilots_1_data_set.mat');
        case 2
            chan_save_file = 'test_dataset_v3_20_pilots_2_data_set.hdf5';
            load('prec_comb_sig_20_pilots_2_data_set.mat');
        case 3
            chan_save_file = 'test_dataset_v3_20_pilots_3_data_set.hdf5';
            load('prec_comb_sig_20_pilots_3_data_set.mat');
        case 4
            chan_save_file = 'test_dataset_v3_40_pilots_1_data_set.hdf5';
            load('prec_comb_sig_40_pilots_1_data_set.mat');
        case 5
            chan_save_file = 'test_dataset_v3_40_pilots_2_data_set.hdf5';
            load('prec_comb_sig_40_pilots_2_data_set.mat');
        case 6
            chan_save_file = 'test_dataset_v3_40_pilots_3_data_set.hdf5';
            load('prec_comb_sig_40_pilots_3_data_set.mat');
        case 7
            chan_save_file = 'test_dataset_v3_80_pilots_1_data_set.hdf5';
            load('prec_comb_sig_80_pilots_1_data_set.mat');
        case 8
            chan_save_file = 'test_dataset_v3_80_pilots_2_data_set.hdf5';
            load('prec_comb_sig_80_pilots_2_data_set.mat');
        case 9
            chan_save_file = 'test_dataset_v3_80_pilots_3_data_set.hdf5';
            load('prec_comb_sig_80_pilots_3_data_set.mat');
    end
    
    channel_data_imag = h5read(chan_save_file,'/training_data_imag');
    channel_data_real = h5read(chan_save_file,'/training_data_real');
    channel_data_complex = channel_data_real + 1i*channel_data_imag;
    
    Nc = size(channel_data_complex,1);
    Ntrain = size(channel_data_complex,2)/Lr;
    estimated_final_result = zeros(Nc,Nr,Nt,Nfft);
    
    Phi=zeros(Ntrain*Lr,Nt*Nr);%Initialize measurement matrix Phi in [1,(10)] of size LrxNtNr.
    %% precoders and combiners generation
    rng(1);
    
    Ftr = Ftr_save;
    Wtr = Wtr_save(:,1:Ntrain*Lr);
    for i=1:Ntrain
        Phi((i-1)*Lr+(1:Lr),:)=kron(signal_save(:,i).'*Ftr(:,(i-1)*Lt+(1:Lt)).',Wtr(:,(i-1)*Lr+(1:Lr))');% Generate Phi in (13)
    end
    
    %%
    Cw = zeros(Ntrain*Lr, Ntrain*Lr);
    
    for i=1:Ntrain
        Wtr_temp = Wtr(:,(i-1)*Lr+1:i*Lr);
        Cw((i-1)*Lr+1:i*Lr,(i-1)*Lr+1:i*Lr) = Wtr_temp'*Wtr_temp;
    end
    
    D_w = chol(Cw);
    
    r = zeros(Ntrain*Lr,Nfft);%Initialize RX training symbols for one channel
    R = zeros(Nc,Ntrain*Lr,Nfft);% Initialize RX training synbols for all the channels
    
    for num_channels = 1:Nc
        
        tic;
        Gt = 256;
        Gr = 256;
        At_1 = 1/sqrt(Nt) * exp(pi*1j * (0:Nt-1)' * (1-2/Gt*(0:Gt-1)));
        Ar_1 = 1/sqrt(Nr) * exp(pi*1j * (0:Nr-1)' * (1-2/Gr*(0:Gr-1)));
        psiMat_1 = kron(conj(At_1),Ar_1);
        At_11 = At_1;
        Ar_11 = Ar_1;
        fprintf('\tSW-OMP START ');
        A_1 = Phi * psiMat_1;
        r = squeeze(channel_data_complex(num_channels,:,:));
        % Noise variance computation
        r1 = D_w'\r;
        AvgRxPower = norm(reshape(r1,[],1))^2/(Nfft*Ntrain*Lr);
        AvgRxPower = AvgRxPower - abs(mean(reshape(r1,[],1)))^2;
        AvgNoisePower_Est = max(0,AvgRxPower-1);
        
        [estimated_Hk,estimated_Hsparse,psiMat_cols,~,AoD_Indices,AoA_Indices,Gt1,Gr1,At_1,Ar_1,At_cols,Ar_cols] ...
            = sw_omp_algorithm_modified_1(Nfft,Ntrain,AvgNoisePower_Est,D_w,Phi,A_1,At_1,Ar_1,r,psiMat_1,Lr,Gt,Gr,Nt,Nr);
        noiseTerm = D_w'\(r-Phi*estimated_Hk);
        noiseTerm = reshape(noiseTerm,[],1);
        AvgNoisePower_Est = real(noiseTerm'*noiseTerm / length(noiseTerm));
        AvgNoisePower_Est = AvgNoisePower_Est-abs(mean(reshape(noiseTerm,[],1)))^2;
        
        A1 = D_w'\Phi*psiMat_cols;
        [hVec, N_Iter, gamma_vec] = MSBL(A1, r1, AvgNoisePower_Est, MSBL_CONVERGENCE_TOL, MSBL_MAX_ITER);%, GainVec_DFT_Orig);     % Last input argument is dummy
        if Ntrain==20
            Gamma_threshold = AvgNoisePower_Est+5;
        end
        if Ntrain==40
            Gamma_threshold = AvgNoisePower_Est+1;
        end
        if Ntrain==80
            Gamma_threshold = AvgNoisePower_Est+0.1;
        end
        hVec(gamma_vec<Gamma_threshold,:)=0;
        
        estimated_Hk_MSBL = psiMat_cols*hVec;
        noiseTerm = D_w'\(r-Phi*estimated_Hk_MSBL);
        noiseTerm = reshape(noiseTerm,[],1);
        AvgNoisePower_Est = real(noiseTerm'*noiseTerm / length(noiseTerm));
        AvgNoisePower_Est = AvgNoisePower_Est-abs(mean(reshape(noiseTerm,[],1)))^2;
        AvgNoisePower_Est_dB = 10*log10(AvgNoisePower_Est);
        
        % Lag domain sparsity
        if Ntrain==20
            if AvgNoisePower_Est_dB<=0.5
                Taps_to_Retain = 8;
            end
            if AvgNoisePower_Est_dB>0.5 && AvgNoisePower_Est_dB<=3.5
                Taps_to_Retain = 7;
            end
            if AvgNoisePower_Est_dB>3.5 && AvgNoisePower_Est_dB<=8.5
                Taps_to_Retain = 6;
            end
            if AvgNoisePower_Est_dB>8.5 && AvgNoisePower_Est_dB<=11.5
                Taps_to_Retain = 5;
            end
            if AvgNoisePower_Est_dB>11.5 && AvgNoisePower_Est_dB<=14.5
                Taps_to_Retain = 4;
            end
            if AvgNoisePower_Est_dB>14.5 && AvgNoisePower_Est_dB<=18.5
                Taps_to_Retain = 3;
            end
            if AvgNoisePower_Est_dB>18.5
                Taps_to_Retain = 2;
            end
        end
        
        if Ntrain==40
            if AvgNoisePower_Est_dB<=1.5
                Taps_to_Retain = 9;
            end
            if AvgNoisePower_Est_dB>1.5 && AvgNoisePower_Est_dB<=3.5
                Taps_to_Retain = 8;
            end
            if AvgNoisePower_Est_dB>3.5 && AvgNoisePower_Est_dB<=6.5
                Taps_to_Retain = 7;
            end
            if AvgNoisePower_Est_dB>6.5 && AvgNoisePower_Est_dB<=12.5
                Taps_to_Retain = 6;
            end
            if AvgNoisePower_Est_dB>12.5 && AvgNoisePower_Est_dB<=14.5
                Taps_to_Retain = 5;
            end
            if AvgNoisePower_Est_dB>14.5 && AvgNoisePower_Est_dB<=18.5
                Taps_to_Retain = 4;
            end
            if AvgNoisePower_Est_dB>18.5
                Taps_to_Retain = 3;
            end
        end
        
        if Ntrain==80
            if AvgNoisePower_Est_dB<=1.5
                Taps_to_Retain = 10;
            end
            if AvgNoisePower_Est_dB>1.5 && AvgNoisePower_Est_dB<=4.5
                Taps_to_Retain = 9;
            end
            if AvgNoisePower_Est_dB>4.5 && AvgNoisePower_Est_dB<=7.5
                Taps_to_Retain = 8;
            end
            if AvgNoisePower_Est_dB>7.5 && AvgNoisePower_Est_dB<=11.5
                Taps_to_Retain = 7;
            end
            if AvgNoisePower_Est_dB>11.5 && AvgNoisePower_Est_dB<=15.5
                Taps_to_Retain = 6;
            end
            if AvgNoisePower_Est_dB>15.5 && AvgNoisePower_Est_dB<=18.5
                Taps_to_Retain = 5;
            end
            if AvgNoisePower_Est_dB>18.5
                Taps_to_Retain = 4;
            end
            
        end
        
        HEst_freq_MSBL = reshape(estimated_Hk_MSBL,Nr,Nt,Nfft);
        HEst_time_MSBL = zeros(Nr,Nt,Nfft);
        for iNr=1:Nr
            for iNt = 1:Nt
                HEst_time_MSBL(iNr,iNt,:) = (ifft(HEst_freq_MSBL(iNr,iNt,:),Nfft));
                tempMat1 = squeeze(HEst_time_MSBL(iNr,iNt,:));
                [sortVals,sortIndices] = sort(abs(tempMat1),'descend');
                HEst_time_MSBL(iNr,iNt,sortIndices(Taps_to_Retain+1:end)) = 0;
                HEst_freq_MSBL(iNr,iNt,:) = (fft(HEst_time_MSBL(iNr,iNt,:),Nfft));
            end
        end
        
        estimated_Hk_MSBL = reshape(HEst_freq_MSBL,Nr*Nt,Nfft);
        
        time_SWOMP = toc;
        
        estimated_final_result(num_channels,:,:,:) = reshape(estimated_Hk_MSBL,Nr,Nt,Nfft);
        fprintf('SW-OMP END, num_channels = %s, data set %s\n',num2str(num_channels),num2str(data_set));
        
    end
    
    save(['estimated_channel_test_dataset_',num2str(data_set),'.mat'],'estimated_final_result','-v7.3');
end
