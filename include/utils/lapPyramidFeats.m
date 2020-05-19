%{ 
Author: Deepti Ghadiyaram

Description: Given an input grayscale image, this method computes a
laplacian image and extracts model parameters (GGD shape and variance) and
sample statistics (kurtosis and skewness)

Output: Features extracted on the laplacian image

Dependencies: This method depends on the following methods:
buildLpyr.m (a method in matlabPyrTools)
pyrBand.m (a method in matlabPyrTools)
estimateggdparam.m

%}
function  feat = lapPyramidFeats(imGray)
    addpath('include/matlabPyrTools');
    
    % Extract the first laplacian of the image.
    [pyr pind] = buildLpyr(imGray,4);
    res =  pyrBand(pyr, pind, 1);
    
    % Fit a GGD to the laplacian image.
    [alpha, sigma] = estimateggdparam(res(:));
    
    % Extract the model parameters (shape, sigma) and sample statistics
    % (kurtosis and skewness)
    feat = [alpha sigma skewness(res(:)) kurtosis(res(:))];
end