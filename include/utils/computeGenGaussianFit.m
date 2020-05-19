%{ 
Author: Deepti Ghadiyaram

Description: Given an input array, this method computes the goodness of
gaussian fit of I

Input: a MXN array I

Output: Goodness of the generalized gaussian fit of I
%}
function ifit = computeGenGaussianFit(I)
    [count,bins] = hist(I(:),100);
    
    [gbeta,gsigma] =  estimateggdparam(bins);
    galpha= gsigma*sqrt(gamma(1./gbeta)./gamma(3./gbeta));
    
    T1=gbeta/(2*galpha*gamma(1./gbeta));
    T2=(abs(bins)./galpha).^gbeta;
    gProb = T1.*exp(-T2);
    
    gProb = gProb./sum(gProb);
    count = count./sum(count);
    
    ifit=sum(((gProb-count).*(gProb-count)))/(gsigma.*gsigma);
end   