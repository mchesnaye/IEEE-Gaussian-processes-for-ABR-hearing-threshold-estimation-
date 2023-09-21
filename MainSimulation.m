%% 
% Code to run simulations for the GP. This code was run on the Iridis
% supercomputer at Southampton university. 
clear all         
close all         
clc               

load AR_Models_SleepStil        % auto-regressive models for generating noise
load ABRTemplate                % ABR template for simulating a response

% filters
fs             = 5000;                          % sampling rate
[blow, alow]   = butter(3, 1500*2/fs, 'low');   % low pass
[bhigh, ahigh] = butter(3, 30*2/fs, 'high');    % high pass

% parameters
ELen        = 106;                  % samples per epoch
TailN       = 50;                   % to be truncated; filter effects
BootReps    = 2000;                 % number of bootstrap replicates
MaxWin      = [20:40; 30:50; 40:60; 50:70; 60:80];      % PTTa search windows for locating the peak
MinWin      = [40:60; 50:70; 60:80; 70:90; 80:100];     % PTTa search windows for locating the trough

% GP parameters
dBSpace     = -10:90;                       % all test and prediction locations for the GP
LT          = length(dBSpace);              % 
AddN        = 500;                          % GP is re-computed every 500 epochs
Amp_Targets = [0.5, 0.3, 0.2, 0.15, 0.1];   % PTTa targets for the GP to locate
Amp_CI      = [0.2, 0.1, 0.1, 0.1, nan];    % required level of certainty before a target is deemed located
AllSTDVal   = 0.03:0.005:0.1;               % required level of certainty for the 5th PTTa target (i.e., for the 0.1 uV target)
Max_Std 	= 1.25 / norminv(0.999);        % the scale parameter s     
SigScale    = Max_Std^2;

% Simulation starts here
for STDi = 1:length(AllSTDVal)          % evaluate performance for different stopping citeria
    Amp_CI(5)   = AllSTDVal(STDi);
    NumTests    = 10000;            	% number of tests to carry out       
    for ti=1:NumTests
        
        % % %
        % initialise parameters for test ti
        % % %
        TotN_PerLvl     = zeros(1, LT );            % number of epochs tested at each stimulus level
        P2PDat          = zeros(1, LT ) * nan;      % the observed PTTa values
        P2PDat_Clean    = zeros(1, LT ) * nan;      % the estiamted noise-free PTTa values
        NoiseVec        = zeros(1, LT ) * nan;      % the variances associated with the estimated noise-free PTTa values
        CI              = 81;                       % the index for the current stimulus level, here equal to dBSpace(CI) = 70
        for dbi=1:length(dBSpace)
            ObservedData{dbi}	= [];                       % all observed data 
            BootCA{dbi}         = zeros(BootReps, ELen);    % bootstrapped data
        end
        
        % noise for this test
        Rnd1        = ceil( rand(1) * size(AllARModels,1) );        % random index
        Rnd2        = ceil( rand(1) * length( AllSTD ) );           % random index
        ARp         = AllARModels(Rnd1,:);                          % select a random AR model
        NoiseSTD	= AllSTD(Rnd2);                                 % select a random noise variance
        
        % generate amplitude growth function
        HT      = 0 + round( rand(1)*70 );          % select a random hearing threshold 
        AMax	= 0.75 + rand(1)*0.5;               % and a random AMax value
        if rand(1) > 0.5        
            Sensorineural = true;                   % randomly choose conductive or sensorineural hearing loss
        else
            Sensorineural = false;
        end
        Amp_Full	= GenerateGrowthCurve(HT, AMax, Sensorineural);    % generate growth curve
        Amp         = Amp_Full(1:LT);
        
        % % % % % % % % % % % % % %
        % % % testing starts here %
        % % % % % % % % % % % % % %
        EstimatingAudiogram = true;
        while EstimatingAudiogram
        
            % generate noise
            Noise       = filter(1, ARp, randn(1,((AddN + 2*TailN)*ELen)) );    % generate noise        
            Noise       = filtfilt(blow, alow, Noise);                          % filter noise
            Noise       = filtfilt(bhigh, ahigh, Noise);                        % filter noise
            Noise       = (Noise ./ std(Noise)) * NoiseSTD;                     % scale noise
            Noise       = Noise( ((ELen*TailN)+1):(end - ELen*TailN) );         % discard potential filter effects
            NoiseEpochs	= reshape(Noise', [ELen,AddN])';                        % restructure into an ensemble

            % add ABR 
            ThisTemplate	= Template(1:ELen) .* Amp(CI);                      % rescale template
            ABRMat       	= repmat(  ThisTemplate, [AddN, 1]);                % generate an ensemble of ABRs
            ThisData      	= NoiseEpochs + ABRMat;                             % add ABR to noise

            % store 
            ObservedData{CI}	= [ObservedData{CI}; ThisData];                 % store
            TotN_PerLvl(CI)     = size(ObservedData{CI},1);                     % update sample size

            % P2Pa
            CA          = mean( ObservedData{CI} );             % coherent average
            P2PDat(CI)	= ComputePTTa(CA, MaxWin, MinWin);      % 

            % bootstrap data and estimate unbiased PTTa value
            Tot         = TotN_PerLvl(CI);                                          % number of epochs to bootstrap
            BootEpInd	= ((Tot-AddN+1):Tot);                                       % epoch indices
            Epochs_Boot = ObservedData{CI}(BootEpInd,:) - repmat(CA, [AddN, 1]);    % remove coherent average
            BootRec     = reshape( Epochs_Boot', [1, ELen * AddN] );                % restructure to continuous recording
            RecLen      = length(BootRec);                                          %
            BootDat     = zeros(AddN, ELen);                                        % initialize
            for br=1:BootReps
                StartPos = ceil( rand(1,AddN) * ( RecLen - ELen - 1 ) );            % random trigger locations
                for ei=1:AddN
                    BootDat(ei,:) = BootRec( StartPos(ei):(StartPos(ei)+ELen-1) );  % randomly sampled blocks of EEG
                end
                BootDat((1:2:end),:) = BootDat((1:2:end),:) * -1;                   % invert every other epoch to increase random variation
                % coherent averages
                BootCA{CI}(br,:) = ...
                    ( (BootCA{CI}(br,:) * (Tot - AddN)) + (mean(BootDat) * AddN) ) / Tot;   % compute bootstrapped coherent averages
            end

            % % % % % % % % % % % % % % % % % % % % % %
            % % % find most likely unbiased PTTa  % % % 
            % % % % % % % % % % % % % % % % % % % % % %

            clear AllPD_Right AllLL_Right AllMed AllLL_Left

            % 0 location
            P2PDistr0 = zeros(1, BootReps);
            for br=1:BootReps
                P2PDistr0(br) = ComputePTTa( BootCA{CI}(br,:), MaxWin, MinWin);     % distribution of PTTa values for no response condition
            end
            PD_0            = fitdist(P2PDistr0','Kernel');     % fit a kernel
            LL_0            = pdf( PD_0, P2PDat(CI) );          % compute likelihood that the observed PTTa value (given by P2PDat(CI) ) arose under the no stimulus condition
            Med0            = median( PD_0 );                   % median value under no stimulus condition
            AllPD_Right{1}  = PD_0;                             % 
            AllLL_Right(1)	= LL_0;                             % 

            % determine range of the posterior distribution over unbiased PTTa values
            MaxLoop     = true;             %
            ScaleInc    = 0.25;             %
            MaxScale	= 0;                % the boundary for the range of the distribution
            while MaxLoop
                MaxScale	= MaxScale + ScaleInc;      % update boundary
                ThisT       = Template * MaxScale;      % get template 
                for br=1:BootReps
                    CA_B        = BootCA{CI}(br,:) + ThisT(1:ELen);
                    P2PaB(br)	= ComputePTTa(CA_B, MaxWin, MinWin);    % compute distribution of biased PTTa values under ABR template "ThisT"
                end
                % check right tail
                if min( P2PaB ) > (1.5*P2PDat(CI))                      % boundary criteria
                    MaxLoop = false;
                end
                % check left tail
                if ~MaxLoop
                    Tmp_PD      = fitdist( P2PaB' , 'Kernel' );         % fit distribution
                    ThisMed     = median( Tmp_PD );
                    P2P_Shift  	= P2PDat(CI) + 2 * ( ThisMed - Med0 );  % shift distribution to obtain expected distribution for negative biased PTTa values
                    LeftLL      = pdf( Tmp_PD, P2P_Shift );
                    if LeftLL > eps
                        MaxLoop = true;
                    end
                end
            end

            % generate posterior 
            PostRes         = MaxScale / 100;           
            RightScaleVal   = 0:PostRes:MaxScale;       % positive PTTa values along which the posterior will be generated
            for ai=2:length(RightScaleVal)
                ThisT = Template * RightScaleVal(ai);
                for br=1:BootReps
                    CA_B        = BootCA{CI}(br,:) + ThisT(1:ELen);
                    P2PaB(br)	= ComputePTTa(CA_B, MaxWin, MinWin);        % distribution of biased PTTa values
                end
                % right tail
                AllPD_Right{ai} = fitdist( P2PaB' , 'Kernel' );
                AllLL_Right(ai) = pdf( AllPD_Right{ai}, P2PDat(CI) );
                AllMed(ai)      = median( AllPD_Right{ai} );
                % left tail
                P2P_Shift           = P2PDat(CI) + 2 * ( AllMed(ai) - Med0 );   % shifted distribution for negative biased PTTa values
                AllLL_Left(ai-1)	= pdf( AllPD_Right{ai}, P2P_Shift );
            end
            Posterior           = [fliplr(AllLL_Left), AllLL_Right];                            % the full posterior
            AllScaleVal         = [fliplr( RightScaleVal(2:end)*-1 ), RightScaleVal];           % the axis values along which the full psoterior is defined. 
            P2PDat_Clean(CI)	= AllScaleVal( find( Posterior == max(Posterior) ) );           % most likely unbiased PTTa value

            % estimate variance
            ThisCDF         = zeros(1, length(Posterior) );         % initialize cumulative PDF
            Posterior_Norm  = Posterior ./ sum(Posterior);          % normalize area
            for ai=1:length(Posterior_Norm)
                ThisCDF(ai) = sum(Posterior_Norm(1:ai));            % construct CDF
            end
            AMin_1          = min( find( abs( ThisCDF - 0.1587) == min(abs( ThisCDF - 0.1587)) ) );  	% index associated with the 0.1587 percentile 
            AMin_2          = max( find( abs( ThisCDF - 0.8413) == min(abs( ThisCDF - 0.8413)) ) );     % index associated with the 0.8413 percentile 
            AMin_M          = min( find( abs( ThisCDF - 0.5) == min(abs( ThisCDF - 0.5)) ) );           % index associated with the 0.5 percentile 
            STD1            = abs( AllScaleVal(AMin_M) - AllScaleVal(AMin_2) );         % standard deviation estimate 1
            STD2            = abs( AllScaleVal(AMin_M) - AllScaleVal(AMin_1) );         % standard deviation estimate 2
            STD_Est         = ( STD1 + STD2 ) / 2;                                      % final standard deviation estimate
            NoiseVec( CI )	= STD_Est^2;                                                % final variance estimate

            % % % % % % % % % % % % % % % % % % % % % % % % % % %
            % % % % % maximum likelihood estimation for the GP  %
            % % % % % % % % % % % % % % % % % % % % % % % % % % %

            % observations and observation distances
            O       = [];       % observed unbiased PTTa values
            dB_O    = [];       % corresponding stimulus levels
            Noise_O	= [];       % corresponding noise levels
            for dbi=1:LT
                if ~isnan( P2PDat_Clean(dbi) )
                    O(end+1)        = P2PDat_Clean(dbi);    
                    dB_O(end+1)     = dBSpace( dbi );
                    Noise_O(end+1)	= NoiseVec( dbi );
                end
            end
            OLen            = length(O);
            dB_Distances_O	= (dB_O - dB_O').^2;        % distances in stimulus levels between observations

            % maximise likelihood
            LendB_Val = 1000:50:2000;                 	% length scale (for the covariance function) to evaluate 
            for dbi=1:length(LendB_Val)
                LendB   = LendB_Val(dbi);           
                Cov_Exp	= SigScale * exp( -( dB_Distances_O / LendB )  );   % get covariance matrix for prediction locations
                Cov_O   = Cov_Exp + eye(OLen)*eps*1000;                     % get covariance matrix for test locations
                Cov_O   = Cov_O + eye(OLen) .* Noise_O;                     % add noise component
                % Posterior
                L       = chol(Cov_O)';
                Alpha   = L' \ (L \ O');             % = inv(Cov_O)*y
                % Likelihood observations
                LLSum(dbi) = -0.5 * O * Alpha - sum( log( diag(L) ) );      % likelihood
            end
            % most likely length scale
            TmpI    = find( LLSum == max(LLSum) );
            LendB	= LendB_Val( TmpI(1) );

            % % % % % % % % % % % % % % % % % % % %
            % % % % % Generate GP posterior % % % %
            % % % % % % % % % % % % % % % % % % % %

            % prediction distances
            Cp = 0;
            for dbi=1:length(dBSpace)
                Cp          = Cp + 1;
                dB_P(Cp)    = dBSpace( dbi );   
            end     
            dB_Distances_P	= (dB_P - dB_P').^2;    % distances between prediction locations
            dB_Distances_PO	= (dB_P - dB_O').^2;    % distances between test and prediction locations
            % cov matrices
            Cov_O       = SigScale .* exp( -( dB_Distances_O ./ LendB )  );         % covariance matrix for test locations
            Cov_P       = SigScale .* exp( -( dB_Distances_P ./ LendB )  );         % covariance matrix for prediction locations
            Cov_PO      = SigScale .* exp( -( dB_Distances_PO ./ LendB )  );        % covariance matrix between test and prediction locations
            % add noise
            Cov_O       = Cov_O + eye(OLen)*eps*1000;                               % add minimal noise component
            Cov_O       = Cov_O + eye(OLen) .* Noise_O;                             % add noise component

            % prior and posterior 
            Prior       = zeros( 1, LT );                                       % prior mean
            PostMu_MLE	= Prior + ( Cov_PO' * inv(Cov_O) * ( O - 0 )' )';       % posterior mean
            PostCov_MLE	= Cov_P - Cov_PO'*inv(Cov_O)*Cov_PO;                    % posterior covariance
            DiagSTD_MLE	= sqrt(diag(PostCov_MLE))';                             % main diagonal of the posterior covariances


            % % % % % % % % % % % % % % % % % % % % % % %
            % plots - uncomment code below to visalize  %
            % % % % % % % % % % % % % % % % % % % % % % %
%             TmpInd = find( ~isnan(P2PDat_Clean) == 1 );
%             NSTD = norminv(0.999); 
%             close all
%             figure('units','normalized','outerposition',[0 0 1 1])
%             subplot(2,2,1);hold on
% 
%             % Posterior
%             plot( dBSpace, PostMu_MLE, '-.', 'color', 'b', 'LineWidth', 3 )
%             plot( dBSpace, PostMu_MLE - NSTD * DiagSTD_MLE, '-', 'color', 'b', 'LineWidth', 1 )
%             plot( dBSpace, PostMu_MLE + NSTD * DiagSTD_MLE, '-', 'color', 'b', 'LineWidth', 1 )
%             % data points
%             plot( dBSpace(TmpInd), P2PDat_Clean(TmpInd),'*', 'color', [0.8510    0.3255    0.0980], 'LineWidth', 3 )
%             % axis
%             set( gca, 'FontSize', 12, 'LineWidth', 2, 'XTick', -10:20:90, 'YTick', -1.5:0.5:1.5 )
%             axis([-10 90 -1.5 2])
%             grid on
%             ylabel({'P2P amplitude (uV)'})
%             xlabel('dB level')

            % % % % % % % % % % % % % % %
            % % % chose next level  % % %
            % % % % % % % % % % % % % % %

            % reset
            CI      = [];
            AmpVal  = 0:0.01:1;

            % most likely amplitude at max dB level
            for ai=1:length(AmpVal)
                LL_Amp_LT( ai ) = normpdf( AmpVal(ai), PostMu_MLE(LT), DiagSTD_MLE(LT) );
            end
            MaxLL_LT_Amp = AmpVal( min(find( LL_Amp_LT == max(LL_Amp_LT) )) );

            % Update likelihoods for all targets
            for ai=1:length( Amp_Targets )
                for dbi=1:LT
                    LL_Target_Amps(ai,dbi) = normpdf( Amp_Targets(ai), PostMu_MLE(dbi), DiagSTD_MLE(dbi) );
                end
            end

            % most likely dB locations for targets
            dBRES = 1;
            for ai=1:length(Amp_Targets)
                TargetLoc(ai) = max( find(LL_Target_Amps(ai,:) == max( LL_Target_Amps(ai,:) )) );
            end
            TargetLoc = round( (TargetLoc - 1) ./ dBRES) * dBRES + 1;  

            % check boundary to facilitate monotonicity
            for ai=1:length(Amp_Targets)
                if MaxLL_LT_Amp <= Amp_Targets( ai ) 
                    TargetLoc(ai) = LT;
                end
            end

            % STD associated with target locations
            for ai=1:length(Amp_Targets)
                Associated_STD(ai) = DiagSTD_MLE( TargetLoc(ai) );
            end

            % next level
            for ai=1:length(Amp_Targets)
                if Associated_STD( ai ) >= Amp_CI(ai)
                    CI              = TargetLoc( ai );     
                    FoundNextLvl    = true;
                    break;
                end
            end

            % stop criterion
            if isempty(CI)          
               EstimatingAudiogram = false;
            else
                CurrentLevel = dBSpace(CI);
            end
           
        end
        
        % estimated hearing threshold
        for dbi=1:LT
            HT_LL(dbi) = normpdf( 0,PostMu_MLE(STDi,dbi), DiagSTD_MLE(STDi,dbi) );
        end
        TmpI    = find(HT_LL==max(HT_LL));
        HTEst       = dBSpace(TmpI)         % estimated threshold
        HT_Error    = HTEst - HT            % dB estimation error
        
    end
end
      

        
        

