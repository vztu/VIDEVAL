function cwssim = cwssim_index_new(band1, band2, K)
%========================================================================
%CW-SSIM Index, Version 1.0
%Copyright(c) 2010  Zhou Wang, Mehul Sampat and Alan Bovik
%All Rights Reserved.
%----------------------------------------------------------------------
%Permission to use, copy, or modify this software and its documentation
%for educational and research purposes only and without fee is hereby
%granted, provided that this copyright notice and the original authors'
%names appear on all copies and supporting documentation. This program
%shall not be used, rewritten, or adapted as the basis of a commercial
%software or hardware product without first obtaining permission of the
%authors. The authors make no representations about the suitability of
%this software for any purpose. It is provided "as is" without express
%or implied warranty.
%----------------------------------------------------------------------
%
%This is an implementation of the algorithm for calculating the Complex-Wavelet 
%Structural SIMilarity (CW-SSIM) index between two images. Please refer
%to the following paper:
%
% M. P. Sampat, Z. Wang, S. Gupta, A. C. Bovik, M. K. Markey. 
% "Complex Wavelet Structural Similarity: A New Image Similarity Index", 
% IEEE Transactions on Image Processing, 18(11), 2385-401, 2009.
%
% *** Important code dependencies: ***
% This code requires the "matlabPyrTools" package developed by Prof. Eero Simoncelli. 
% This package can be downloaded from: http://www.cns.nyu.edu/~lcv/software.php 
%
% Kindly report any suggestions or corrections to mehul.sampat@ieee.org
%
%----------------------------------------------------------------------
%
% Input :::
%            (1) img1......the first image being compared
%            (2) img2......the second image being compared
%            (3) level......the number of levels to used in the complex steerable pyramid decomposition   
%            (4) or..........the number of orientations to be used in the complex steerable pyramid decomposition     
%            (5) guardb...this parameter is used to control how much is discarded from the four image boundaries. 
%            (6) K...........the constant in the CWSSIM index formula (see the above reference) default value: K=0
%
% Output ::: 
%            (1) cwssim...the CWSSIM index value between 2 images. If img1 = img2, then cwssim = 1.
%
% Example Usage: Given 2 test images img1 and img2
%
% cwssim = cwssim_index(img1, img2,6,16,0,0);
%
% See the results: "cwssim" gives the CW-SSIM index value
%========================================================================

% [pyr1, pind] = buildSCFpyr(img1, level, or-1);%........% decompose img1 using a complex steerable pyramid decomposition
% [pyr2, pind] = buildSCFpyr(img2, level, or-1);%........% decompose img2 using a complex steerable pyramid decomposition

winsize = 7;
window = ones(7);%..............................................% The CW-SSIM indices are computed locally using a sliding 
%                                                                         % 7-by-7 window that moves across each wavelet subband.

window = window./sum(sum(window));%................% normalize the window

% gb = guardb/(2^(level-1));%...................................% the gb parameter is used to control how much is discarded from the four image boundaries. 

% s = pind((level-1)*or+2, :);
s = size(band2);
w_sigma = s(1)/4;
% w = fspecial('gaussian', s-winsize+1, s(1)/4);%........% The CW-SSIM index map is combined into a scalar similarity measure using a
                                                                           % weighted summation.The weighting function is obtained using a Gaussian 
                                                                           % profile with a standard deviation equaling a quarter of the image size at 
                                                                           % finest level of pyramid decomposition.
w = fspecial('gaussian', s-winsize+1, w_sigma);
% for i=1:or
%    bandind = i+(level-1)*or+1;
%    band1 = pyrBand(pyr1, pind, bandind);%............% Access a subband from a pyramid (see help pyrBand)
%    band2 = pyrBand(pyr2, pind, bandind);%............% Access a subband from a pyramid (see help pyrBand)
%    band1 = band1(gb+1:end-gb, gb+1:end-gb);
%    band2 = band2(gb+1:end-gb, gb+1:end-gb);
%    band1 = imresize(band1,size(band2));
   corr = band1.*conj(band2);
   varr = abs(band1).^2 + abs(band2).^2;
   corr_band = filter2(window, corr, 'valid');%.........% The CW-SSIM indices are computed locally using a sliding 7-by-7 
   varr_band = filter2(window, varr, 'valid');%.........% window that moves across each wavelet subband.

   cssim_map = ...
   (2*abs(corr_band) + K)./(varr_band + K);%........% The purpose of the small constant K is mainly to improve the robustness of 
                                                                           % the CW-SSIM measure when the local signal to noise ratios are low.

   cwssim = sum(sum(cssim_map.*w));%......% The CW-SSIM index map is combined into a scalar similarity measure using a
                                                                           % weighted summation.
% end

% cwssim = mean(band_cssim);%...............................% value is 1 for two identical images.