%{ 
Author: Deepti Ghadiyaram

Description: Given an input gray scale image in the form of a 2-D array
(imGray), this method does the following:

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

function feat = neighboringPairProductFeats(imGray)
    
    if(size(imGray,3) ~=1)
        warning('The input to fourNeighborhoodFeats is not a double array of gray scale image. Constructing this image now');
        imGray = double(rgb2gray(imGray));
    end
    
    % Applying divisive normalization operation on the given gray scale image.
    [structdis,~] = divisiveNormalization(imGray);
    
    % The four neighborhood maps that would be constructed [1]
    shifts = [ 0 1;1 0 ; 1 1; -1 1];

    % Constructing 4 neighborhood maps and extracting model features (AGGD
    % parameters) and sample features (kurtosis and skrewness) 
    modelFeat = [];
    sampleFeat = [];
    
    for itr_shift =1:4
        % Construct the product neighborhood map.
        shifted_structdis = circshift(structdis,shifts(itr_shift,:));
        pair = structdis(:).*shifted_structdis(:); % Element wise product.
        
        % Fit an AGGD and extract its parameters.
        [alpha, leftstd, rightstd] = estimateaggdparam(pair);
        const = (sqrt(gamma(1/alpha))/sqrt(gamma(3/alpha)));
        meanparam = (rightstd-leftstd)*(gamma(2/alpha)/gamma(1/alpha))*const;
        
        % Aggregate the model parameters
        modelFeat =  [modelFeat alpha meanparam leftstd^2 rightstd^2];
        % Aggregate the sample parameters.
        sampleFeat = [sampleFeat skewness(pair(:)) kurtosis(pair(:))];
    end
    
    feat = [modelFeat sampleFeat];
end