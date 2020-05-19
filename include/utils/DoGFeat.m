%{ 
Author: Deepti Ghadiyaram

Description: Given an input RGB image, this method extracts the 564 FRIQUEE
features proposed in [1] in three different color spaces.

1. We first convert the given input image into different color spaces -
LAB, LMS, and HSI.

2. We then extract the required components from each of these color spaces
eg: Hue and Saturation components, M and S components.

3. We also compute the chroma map from the A and B components of the image
in LAB color space.

Input: a MXNX3 array upon reading any image.

Output: Features extracted in luminance, chroma, LMS color components of the
image.

Dependencies: This method depends on the following methods:
computeSigmaMap.m
divisiveNormalization.m
%}
function feat = DoGFeat(imGray)
    
    %Get the sigma map of the image.
    sigmaMap = computeSigmaMap(imGray);
    
    % Construct two Gaussian windows where sig2 = 1.5*sig1;
    k=1.5;
    
    window1 = fspecial('gaussian',7,7/6);
    window1 = window1/sum(sum(window1));

    window2 = fspecial('gaussian',7,7*k/6);
    window2 = window2/sum(sum(window2));
    
    %Compute DOG of the sigmaMap
    DoGSigma= filter2((window1-window2), sigmaMap, 'same');
    
    %Compute the sigmaMap of DoGSigma
    DoGSigma1 = computeSigmaMap(DoGSigma);
    
    %DNT of DOG of sigma.
    divNormDoGSigma = divisiveNormalization(DoGSigma);

    %shape, variance, skewness.
    [alpha,lsigma] = estimateggdparam(divNormDoGSigma(:));
    
    % Apply DNT on DoGSigma1
    divNormDoGSigma1 = divisiveNormalization(DoGSigma1);
    
    % The model features are the GGD parameters of divNormDoGSigma (DNT of
    % dogSigma)
    modelFeats = [alpha lsigma];
    
    % The sample statistical features are extracted from both
    % divNormDoGSigma and divNormDoGSigma1
    sampleFeats = [skewness(divNormDoGSigma(:)) kurtosis(divNormDoGSigma(:)) skewness(divNormDoGSigma1(:)) kurtosis(divNormDoGSigma1(:))];
    
    feat = [modelFeats sampleFeats];
end