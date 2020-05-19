%{ 
Author: Deepti Ghadiyaram

Description: Given an input image in LMS color space, this method extracts
the color opponent features described in 'LMS Feature Maps' section of [1]

Input: a MXNX3 array of LMS map 

Output: Features extracted from the color opponent maps constructed in LMS color space(as described in the section
LMS Feature Maps in [1])

Dependencies: This method depends on the following methods:
    divisiveNormalization.m
    estimateaggdparam.m

Reference:
[1] D. Ghadiyaram and A.C. Bovik, "Perceptual Quality Prediction on Authentically Distorted Images Using a
Bag of Features Approach," http://arxiv.org/abs/1609.04757
%}

function feat = lmsColorOpponentFeats(lms)
    % Extract the three channel maps (L,M, and S)
    l = double(lms(:,:,1))+1; 
    m = double(lms(:,:,2))+1; 
    s = double(lms(:,:,3))+1; 
    
    % Apply divisive normalization on log of these channel maps.
    dL = divisiveNormalization(log(l));
    dM = divisiveNormalization(log(m));
    dS = divisiveNormalization(log(s));
    
    % Constructing the BY-color opponency map.
    dBY = (dL+dM-2*dS)./sqrt(6);
    
    % Extracting model parameters (derived from AGGD fit) and the sample
    % statistical features (kurtosis and skewness)
    [beta2,lsigma2,rsigma2] = estimateaggdparam(dBY(:));
    kurt2=kurtosis(dBY(:));
    sk2 = skewness(dBY(:));
    
    % Constructing the RG-color opponency map.
    dRG = (dL-dM)./sqrt(2);
    
    % Extracting model parameters (derived from AGGD fit) and the sample
    % statistical features (kurtosis and skewness)
    [beta3,lsigma3,rsigma3] = estimateaggdparam(dRG(:));
    kurt3=kurtosis(dRG(:));
    sk3 = skewness(dRG(:));
    
    % Aggregating the final set of features from both the opponency maps.
    modelParamsBY = [beta2,lsigma2,rsigma2];
    sampleParamsBY = [kurt2 sk2];
    
    modelParamsRG = [beta3,lsigma3,rsigma3];
    sampleParamsRG = [kurt3 sk3];
    
    feat = [modelParamsRG sampleParamsRG modelParamsBY sampleParamsBY];
end