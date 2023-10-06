%% 
% Code for running the simulations for evaluating the GP in paper: % M. A. Chesnaye, D. M. Simpson, J. Schlittenlacher and S. L. Bell, "Gaussian Processes for hearing threshold estimation using Auditory Brainstem Responses," in IEEE Transactions on Biomedical Engineering, doi: 10.1109/TBME.2023.3318729.
% for questions or comments, please feel free to contact Michael Chesnaye at mchesnaye@gmail.com 
clear all         
close all         
clc               

% load auto-regressive models for emulating the EEG background activity
load ARModels.mat    

% low- and high-pass filter
fs             = 5000;                          % sampling rate
[blow, alow]   = butter(3, 1500*2/fs, 'low');   % low pass
[bhigh, ahigh] = butter(3, 30*2/fs, 'high');    % high pass

% ABR template for simulating a response
load ABRTemplate.mat       
Template = mean(All_GCA_151(4:6,:));                         
Template = Template - mean(Template);
Template = Template ./ (max(Template) - min(Template));

% peak-to-trough amplitude (PTTa) search windows
MaxWin = [20:40; 30:50; 40:60; 50:70; 60:80];       % windows for locating the peak
MinWin = [40:60; 50:70; 60:80; 70:90; 80:100];      % windows for locating the trough

% some parameters
ELen        = 106;              % epoch length in samples
TailN       = 50;               % data to be discarded to prevent filter effects
AddN        = 500;              % Analyse data every 500 epochs
BootReps    = 2000;             % number of bootstrap replicates

% GP parameters 
dBSpace     = -10:90;                       % all test and prediction locations for predictions
LT          = length(dBSpace);              %  
AllSTDVal   = 0.03:0.005:0.1;               % all delta_5 stopping criteria values to evaluate
Amp_Targets = [0.5, 0.3, 0.2, 0.15, 0.1];   % all T_i targets for the GP to locate
Amp_CI      = [0.2, 0.1, 0.1, 0.1, nan];    % the delta_1, delta_2, delta_3, and delta_4 stopping criteria
Max_Std 	= 1.25 / norminv(0.999);        % specify range for the GP prior: 99.9% of PTTa values are assumed to be smaller than 1.25 uV
SigScale    = Max_Std^2;                    % transform to variance to get the scale parameter, denoted by "s" in the paper

% plot GP as data accrues
PLOT = true;

% Note that the following simulations were run on the Iridis supercomputer. 
NumTests = 10000;                   % number of hearing threshold estimation trials to carry out (you may want to reduce this!)   
for ti=1:NumTests
    for STDi = 1:length(AllSTDVal)  % evalutate test performance for different delta_5 stopping criteria     
        
        % initialise
        Amp_CI(5)       = AllSTDVal(STDi);    	% update delta_5 stopping criteria
        TotN_PerLvl     = zeros(1, LT );        % number of epochs tested at each level
        P2PDat          = zeros(1, LT ) * nan;  % array containing all observed (i.e., noisy and biased) PTTa values        
        P2PDat_Clean    = zeros(1, LT ) * nan;  % array containing the transformed (noisy, but unbiased) PTTa values        
        NoiseVec        = zeros(1, LT ) * nan;  % noise (i.e., variance) associated with the unbiased PTTa values
        CI              = 81;                   % the current index (CI) for the next stimulus level to test at, given by dBSpace(CI)
        for dbi=1:length(dBSpace)               %
            ObservedData{dbi}	= [];                       % All observed data
            BootCA{dbi}         = zeros(BootReps, ELen);    % All bootstrapped coherent averages
        end
        
        % noise for this test
        Rnd1        = ceil( rand(1) * size(AllARModels,1) );    % a random index 
        Rnd2        = ceil( rand(1) * length( AllSTD ) );       % another random index
        ARp         = AllARModels(Rnd1,:);                      % select a random set of AR parameters
        NoiseSTD	= AllSTD(Rnd2);                             % select a random noise level for this test
        
        % generate the amplitude growth function for this test
        HT      = 0 + round( rand(1)*70 );                  % random hearing threshold to simulate for this test
        AMax	= 0.75 + rand(1)*0.5;                       % random "A" value, indicating the maximum PTTa value at saturation
        if rand(1) > 0.5                                    % randomly choose sensorineural or conductive hearing loss 
            Sensorineural = true;
        else
            Sensorineural = false;
        end
        Amp_Full	= GenerateAmplitudeCurve(HT, AMax, Sensorineural);	% this function generates the growth function for the parameters specified above
        Amp         = Amp_Full(1:LT);                                   
        
        % the test starts here 
        EstimatingHT = true;
        while EstimatingHT
        
            % generate 500 epochs 
            Noise       = filter(1, ARp, randn(1,((AddN + 2*TailN)*ELen)) );    % filtered GWN to emulate the EEG background activity
            Noise       = filtfilt(blow, alow, Noise);                          % low-pass
            Noise       = filtfilt(bhigh, ahigh, Noise);                        % and high-pass
            Noise       = (Noise ./ std(Noise)) * NoiseSTD;                     % rescale 
            Noise       = Noise( ((ELen*TailN)+1):(end - ELen*TailN) );         % truncate to remove filter effects
            NoiseEpochs	= reshape(Noise', [ELen,AddN])';                        % restructure into an ensemble of epochs
            % add an ABR 
            ThisTemplate	= Template(1:ELen) .* Amp(CI);                      % recale the ABR template in accordance with the growth function at location CI
            ThisData      	= NoiseEpochs + repmat(  ThisTemplate, [AddN, 1]);  % add it to the noise

            % update existing data
            ObservedData{CI}	= [ObservedData{CI}; ThisData];	% add new data
            TotN_PerLvl(CI)     = size(ObservedData{CI},1);     % update ensemble size for stimulus level dBSpace(CI)

            % estimate biased PTTa 
            CA          = mean( ObservedData{CI} );         % coherent average
            P2PDat(CI)	= GetPTTa(CA, MaxWin, MinWin);      % function to estimate the (noisy and biased) PTTa value

            % % % % % % % % % % % % % % % % % % % % % %
            % bootstrap for unbiased PTTa estimation  % 
            % % % % % % % % % % % % % % % % % % % % % %
            clear AllPD_Right AllLL_Right AllMed AllLL_Left                         % reset
            Tot         = TotN_PerLvl(CI);                                          % ensemble size
            BootEpInd	= ((Tot-AddN+1):Tot);                                       % epoch indices to bootstrap from
            Epochs_Boot = ObservedData{CI}(BootEpInd,:) - repmat(CA, [AddN, 1]);    % subtract coherent average from epochs prior to bootstrapping 
            BootRec     = reshape( Epochs_Boot', [1, ELen * AddN] );                % reshape to continuous recording
            RecLen      = length(BootRec);                                          % 
            BootDat     = zeros(AddN, ELen);                                        % initialize
            for br=1:BootReps
                StartPos = ceil( rand(1,AddN) * ( RecLen - ELen - 1 ) );            % random trigger locations
                for ei=1:AddN
                    BootDat(ei,:) = BootRec( StartPos(ei):(StartPos(ei)+ELen-1) );  % resample blocks of EEG of length ELen
                end
                BootDat((1:2:end),:) = BootDat((1:2:end),:) * -1;                   % invert half of the epochs to increase random variation
                % update the bootstrapped coherent averages
                BootCA{CI}(br,:) = ...
                    ( (BootCA{CI}(br,:) * (Tot - AddN)) + (mean(BootDat) * AddN) ) / Tot;      
            end
            
            % First approximate the distribution for "P2PDat(CI)" under the assumption that an ABR is absent, i.e., the null distribution
            P2PDistr0 = zeros(1, BootReps);
            for br=1:BootReps
                P2PDistr0(br) = GetPTTa(BootCA{CI}(br,:), MaxWin, MinWin); 
            end
            PD_0            = fitdist(P2PDistr0','Kernel');     % fit a kernel to the distribution
            LL_0            = pdf( PD_0, P2PDat(CI) );          % likelihood that the observed PTTa value arose under the hypothesis of ABR absent
            Med0            = median( PD_0 );                   % median value of the bootstrapped null distribution
            AllPD_Right{1}  = PD_0;             % store approximated null distribution
            AllLL_Right(1)	= LL_0;             % store the likelihood
            % note that "AllLL_Right(1)" represents the likelihood that
            % "P2PDat(CI)" arose under the assumption that data contained
            % an ABR with a true (noise-free and unbiased) amplitude of 0
            
            % Next, define the range along which the posterior distribution
            % is to be defined (i.e., the distribution over unbiased PTTa values).
            MaxLoop     = true;
            ScaleInc    = 0.25;           
            MaxScale	= 0;                            % the largest potential unbiased PTTa value in the data
            while MaxLoop
                MaxScale	= MaxScale + ScaleInc;      % increase the largest potential unbiased PTTa value
                ThisT       = Template * MaxScale;      % rescale the ABR template accordingly
                for br=1:BootReps
                    CA_B        = BootCA{CI}(br,:) + ThisT(1:ELen);         % add template to the previously bootstrapped (in)coherent averages
                    P2PaB(br)	= GetPTTa(CA_B, MaxWin, MinWin);            % compute distribution of PTTa values under this template
                end
                % check right tail
                if min( P2PaB ) > (1.5*P2PDat(CI))      % if all bootstrapped PTTa values are considerably larger than the observed PTTa value
                    MaxLoop = false;                    % then [-MaxScale, MaxScale] defines the range for the posterior 
                end
                % check left tail                   
                if ~MaxLoop                             % check if a large negative PTTa value is feasible
                    Tmp_PD      = fitdist( P2PaB' , 'Kernel' );         % fit a kernel to the bootstrapped distribution
                    ThisMed     = median( Tmp_PD );                     % find the median value
                    P2P_Shift  	= P2PDat(CI) + 2 * ( ThisMed - Med0 );  % shift the bootstrapped distribution relative to the null distribution
                    LeftLL      = pdf( Tmp_PD, P2P_Shift );             % compute likelihood that the observed (and now shifted) PTTa value arose under the assumption that data contained a negative PTTa value of -MaxScale
                    if LeftLL > eps         % if likelihood is now sufficiently low
                        MaxLoop = true;     % then range needs to be further increased
                    end
                end
            end

            % Now the axis has been defined, the posterior can be generated
            PostRes         = MaxScale / 100;           % posterior is defined across 2*100 + 1 points 
            RightScaleVal   = 0:PostRes:MaxScale;       % the non-negative unbiased PTTa values along which the posterior is defined
            AllScaleVal    	= [fliplr( RightScaleVal(2:end)*-1 ), RightScaleVal];   % the full axis along which the posterior is defined
            for ai=2:length(RightScaleVal)
                ThisT = Template * RightScaleVal(ai);   % rescale template
                for br=1:BootReps
                    CA_B        = BootCA{CI}(br,:) + ThisT(1:ELen);     % add template to previously bootstrapped (in)coherent averages
                    P2PaB(br)	= GetPTTa(CA_B, MaxWin, MinWin);        % approximate the distribution of the (biased and noisy) PTTa values
                end
                % right tail
                AllPD_Right{ai} = fitdist( P2PaB' , 'Kernel' );         % fit a kernel to the distribution
                AllLL_Right(ai) = pdf( AllPD_Right{ai}, P2PDat(CI) );   % compuate likelihood of observing "P2PDat(CI)" under this distribution
                AllMed(ai)      = median( AllPD_Right{ai} );            % compute median value of distribution
                % left tail
                P2P_Shift           = P2PDat(CI) + 2 * ( AllMed(ai) - Med0 );   % shift the observed PTTa value relative to the null distribution
                AllLL_Left(ai-1)	= pdf( AllPD_Right{ai}, P2P_Shift );        % likelihood of observing "P2PDat(CI)" under the assumption that data contained an ABR with a PTTa value equal to -RightScaleVal(ai)
            end
            Posterior           = [fliplr(AllLL_Left), AllLL_Right];                    % posterior distribution over negative and non-negative unbiased PTTa values: plot(AllScaleVal, Posterior)
            P2PDat_Clean(CI)	= AllScaleVal( find( Posterior == max(Posterior) ) );   % most likely unbiased PTTa value

            % estimate variance (noise) associated with "P2PDat_Clean(CI)"
            ThisCDF         = zeros(1, length(Posterior) );     % initialize cumulative distribution
            Posterior_Norm  = Posterior ./ sum(Posterior);      % normalize area of posterior
            for ai=1:length(Posterior_Norm)
                ThisCDF(ai) = sum(Posterior_Norm(1:ai));        % generate cumulative distribution
            end
            AMin_1          = min( find( abs( ThisCDF - 0.1587) == min(abs( ThisCDF - 0.1587)) ) );     % index associated with the 0.1587 percentile
            AMin_2          = max( find( abs( ThisCDF - 0.8413) == min(abs( ThisCDF - 0.8413)) ) );     % index associated with the 0.8413 percentile
            AMin_M          = min( find( abs( ThisCDF - 0.5) == min(abs( ThisCDF - 0.5)) ) );           % index associated with the 0.5 percentile
            STD1            = abs( AllScaleVal(AMin_M) - AllScaleVal(AMin_2) );                         % distance between 0.1587 and 0.5 percentile
            STD2            = abs( AllScaleVal(AMin_M) - AllScaleVal(AMin_1) );                         % distance between 0.8413 and 0.5 percentile
            STD_Est         = ( STD1 + STD2 ) / 2;	% the estimated standard deviation
            NoiseVec( CI )	= STD_Est^2;            % and the estimated variance

            % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
            % maximum likelihood estimation for the length scale parameter  %
            % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
            LendB_Val       = 1000:50:2000;                         % all length scales to evaluate    
            TmpI            = find( ~isnan( P2PDat_Clean ) == 1 );  % 
            O               = P2PDat_Clean( TmpI );                 % observations
            dB_O            = dBSpace( TmpI );                      % test locations
            Noise_O         = NoiseVec( TmpI );                     % noise associated with the observations
            OLen            = length(O);            % number of observations
            dB_Distances_O	= (dB_O - dB_O').^2;    % squared distance between test locations
            for dbi=1:length(LendB_Val)
                LendB   = LendB_Val(dbi);
                Cov_Exp	= SigScale * exp( -( dB_Distances_O / LendB )  );   % covariance matrix for observations
                Cov_O   = Cov_Exp + eye(OLen)*eps*1000;         % add noise    
                Cov_O   = Cov_O + eye(OLen) .* Noise_O;         % add noise
                % Posterior
                L       = chol(Cov_O)';
                Alpha   = L' \ (L \ O');             % = inv(Cov_O) * O;        (but faster)
                % Likelihood observations
                LLSum(dbi) = -0.5 * O * Alpha - sum( log( diag(L) ) );          % equation to generate log likelihood
            end
            % get best parameters
            TmpI    = find( LLSum == max(LLSum) );
            LendB	= LendB_Val( TmpI(1) );             % most likely length scale

            % % % % % % % % % % % % % % % % % % % %
            % % % % % Generate GP posterior % % % % 
            % % % % % % % % % % % % % % % % % % % %

            % prediction locations
            dB_Distances_P	= (dBSpace - dBSpace').^2; 	% squared distance between prediction locations
            dB_Distances_PO	= (dBSpace - dB_O').^2;     % squared distance between prediction and test locations
            Cov_O           = SigScale .* exp( -( dB_Distances_O ./ LendB )  );     % covariance matrix for test locations
            Cov_P           = SigScale .* exp( -( dB_Distances_P ./ LendB )  );     % covariance matrix for prediction locations
            Cov_PO          = SigScale .* exp( -( dB_Distances_PO ./ LendB )  );    % covariance matrix for test and prediction locations
            Cov_O           = Cov_O + eye(OLen)*eps*1000;       % add noise
            Cov_O           = Cov_O + eye(OLen) .* Noise_O;     % add noise
            % % % 
            Prior       = zeros( 1, LT );                                   % prior mean
            PostMu_MLE	= Prior + ( Cov_PO' * inv(Cov_O) * ( O - 0 )' )';   % posterior mean
            PostCov_MLE	= Cov_P - Cov_PO'*inv(Cov_O)*Cov_PO;                % posterior covariance matrix
            DiagSTD_MLE	= sqrt(diag(PostCov_MLE))';                         % sqaure root of the main diagonal

            % % % % %
            % plots %
            % % % % %
            if PLOT
                TmpInd = find( ~isnan(P2PDat_Clean) == 1 );
                NSTD = norminv(0.999); 
                close all
                figure('units','normalized','outerposition',[0 0 1 1])
                subplot(2,2,1);hold on

                % Posterior
                plot( dBSpace, PostMu_MLE, '-.', 'color', 'b', 'LineWidth', 3 )
                plot( dBSpace, PostMu_MLE - NSTD * DiagSTD_MLE, '-', 'color', 'b', 'LineWidth', 1 )
                plot( dBSpace, PostMu_MLE + NSTD * DiagSTD_MLE, '-', 'color', 'b', 'LineWidth', 1 )
                % data points
                plot( dBSpace(TmpInd), P2PDat_Clean(TmpInd),'*', 'color', [0.8510    0.3255    0.0980], 'LineWidth', 3 )
                % axis
                set( gca, 'FontSize', 12, 'LineWidth', 2, 'XTick', -10:20:90, 'YTick', -1.5:0.5:1.5 )
                axis([-10 90 -1.5 2])
                grid on
                ylabel({'P2P amplitude (uV)'})
                xlabel('dB level')
                drawnow()
                pause(0.1)
            end
            
            % % % % % % % % % % % % % % %
            % % % chose next level  % % %
            % % % % % % % % % % % % % % %

            CI = [];           
            
            % most likely amplitude at max dB level
            AmpVal = 0:0.01:1.25; 
            for ai=1:length(AmpVal)
                LL_Amp_LT( ai ) = normpdf( AmpVal(ai), PostMu_MLE(LT), DiagSTD_MLE(LT) );
            end
            MaxLL_LT_Amp = AmpVal( min(find( LL_Amp_LT == max(LL_Amp_LT) )) );  % most likely amplitude at location "LT"

            % find most likely dB levels for evoking ABRs with PTTa values equal to the T_i targets
            for ai=1:length( Amp_Targets )
                for dbi=1:LT
                    LL_Target_Amps(ai,dbi) = normpdf( Amp_Targets(ai), PostMu_MLE(dbi), DiagSTD_MLE(dbi) );
                end
                TargetLoc(ai) = max( find(LL_Target_Amps(ai,:) == max( LL_Target_Amps(ai,:) )) );
            end
 
            % check max test level to facilitate monotonicity
            for ai=1:length(Amp_Targets)
                if MaxLL_LT_Amp <= Amp_Targets( ai )    % if amplitude at max level is smaller than the T_i target, then should test at the max level
                    TargetLoc(ai) = LT;
                end
            end

            % uncertainty (the STD of the GP posterior) associated with the estimated target locations  
            Associated_STD = DiagSTD_MLE( TargetLoc );

            % next level
            for ai=1:length(Amp_Targets)
                if Associated_STD( ai ) >= Amp_CI(ai)   % if there is too much uncertainty, then additional data collection is needed
                    CI              = TargetLoc( ai );     
                    FoundNextLvl    = true;
                    break;
                end
            end

            % stop criterion
            if isempty(CI)                      % then all targets have been located
                EstimatingHT = false;     
            end
        end
        
        % hearing threshold estimate
        for dbi=1:LT
            HT_LL(dbi) = normpdf( 0, PostMu_MLE(dbi), DiagSTD_MLE(dbi) );
        end
        TmpI    = find(HT_LL==max(HT_LL));
        HTEst   = dBSpace(TmpI);
        
        % Store results
        % ...
        
    end
end


