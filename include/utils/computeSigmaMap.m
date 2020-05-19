%{ 
Author: Deepti Ghadiyaram

Description: Given an input gray scale image in the form of a 2-D array
(imGray), this method computes the local standard deviation map (also sigma
field)

1. Constructs 4 pair-wise product neighborhood maps as defined in [1] 

2. Extracts the 16 features that are obtained from fitting an AGGD
(Asymmetric Generalized Gaussian Distribution).

3. Extracts 8 sample statistical features (kurtosis and skewness of each of
the 4 neighborhood maps).

Input: a MXN array of a gray-scale image.

Output: Features extracted on the 4 neighborhood maps.

Dependencies: This method depends on the following methods:
divisiveNormalization.m
estimateaggdparam.m

Reference:
[1] A. Mittal, A. K. Moorthy and A. C. Bovik, "No-reference image quality assessment in the spatial domain," 
IEEE Transactions on Image Processing , vol. 21, no. 12, pp. 4695-4708, December, 2012. 
%}
function sigma = computeSigmaMap(imdist)
    window = fspecial('gaussian',7,7/6);
    window = window/sum(sum(window));
    % Compute a local mean map.
    mu = filter2(window, imdist, 'same');
    
    mu_sq = mu.*mu;
    
    % Compute a sigma map.
    sigma = sqrt(abs(filter2(window, imdist.*imdist, 'same') - mu_sq));
end