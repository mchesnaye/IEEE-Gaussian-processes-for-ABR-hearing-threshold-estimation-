%%
function P2Pa = ComputePTTa(CA, MaxWin, MinWin)
for Wi = 1:size(MaxWin,1)     
    P2PVec(Wi) = max( CA(MaxWin(Wi,:)) ) - min( CA(MinWin(Wi,:)) );
end  
P2Pa = max(P2PVec);