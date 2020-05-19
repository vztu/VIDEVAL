%{ 
Author: Deepti Ghadiyaram

Description: Given an input gray scale image in the form of a 2-D array
(imGray), this method does the following:

1. Applies a Divisive Normalization transform to imGray

2. Fits a Generalized Gaussian Distribution (GGD) and extracts 2 model features (shape, standard deviation of the fit).

Input: a MXN array of a gray-scale image.

Output: Model parameters of the GGD fit.

Dependencies: This method depends on the following methods:
divisiveNormalization.m
estimateggdparam.m
%}

function feat = debiasedNormalizedFeats(imGray)
    
    % Apply DNT
    [imDivNorm,~] = divisiveNormalization(imGray);
    
    % Fit a GGD and extract the shape and standard deviation 
    [alpha,overallstd] = estimateggdparam(imDivNorm(:));
    
    % Model parameters and sample statistical features
    feat = [alpha overallstd kurtosis(imDivNorm(:)) skewness(imDivNorm(:))];
end