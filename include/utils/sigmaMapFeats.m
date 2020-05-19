%{ 
Author: Deepti Ghadiyaram

Description: Given an input gray scale image in the form of a 2-D array
(imGray), this method does the following:

1. Computes the sigma field of this image (the normalization factor in the
divisive normalization operation)

2. Extracts 3 sample statistical features (kurtosis, skewness, and overall mean the sigma field).

Input: a MXN array of a gray-scale image.

Output: Sample statistical features extracted on the sigma field.

Dependencies: This method depends on the following methods:
computeSigmaField.m
%}
function [feat, sigmaMap] = sigmaMapFeats(imGray)
    
    %Extract the sigma field from the image.
    sigmaMap = computeSigmaMap(imGray);
    
    % Compute kurtosis, skewness and average of the sigma field
    feat = [kurtosis(sigmaMap(:)) skewness(sigmaMap(:)) mean2(sigmaMap)];
end