%%
function Amp = GenerateGrowthCurve(HT, A, Sensorineural)

MAX_DB = 90;

% setup cosine parameters
Start	= 1.5*pi + 10*eps; 
End  	= 2*pi;
Range   = End - Start;
Res     = Range / MAX_DB;   
CosInd	= Start:Res:End;
% define growth curves
if Sensorineural==1
    dBRange	= MAX_DB - HT;
    Res     = Range / dBRange;
    dB_HI   = Start:Res:End;
    Amp     = A*cos(dB_HI);
    Amp     = [zeros(1, 10+HT), Amp];
else
    F_NH = A*cos(CosInd);           
    Amp = F_NH( [ 1 + [HT:MAX_DB] ] - HT );
    Amp = [zeros(1, 10+HT), Amp];
end
