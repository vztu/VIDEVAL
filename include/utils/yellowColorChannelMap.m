%{ 
Author: Deepti Ghadiyaram

Description: Given an input RGB image, this method 
1. Computes a yellow color channel map (Y)
2. Applies divisive normalization on Y
3. Extracts sigma map of Y (sigY)
4. Applies divisive normalization on sigY
5. Computes the goodness of Gaussian fit to both Y and sigY 

Input: a MXN array of chroma map (computed for A and B color channels).

Output: Goodness of Gaussian fits of Y and sigY

Dependencies: This method depends on the following methods:
    divisiveNormalization.m
    computeGenGaussianFit.m
   
%}
function feat=yellowColorChannelMap(I)
    if(size(I,3) ~=3)
        error('yellowColorChannelMap method requires rgb components of an image\n');
    end
    
     Ri = double(I(:,:,1));
     Gi = double(I(:,:,2));
     Bi = double(I(:,:,3));

     % Compute the Yellow channel map
     Y = double(((Ri+Gi)./2)- (abs(Ri-Gi)./2) - Bi); % Y-map

     % Apply divisive normalization on yellow map.
     [divNormY,~,~,ySigma]= divisiveNormalization(Y); 
     
     %DNT of sigma of Y
     divNormSigY = divisiveNormalization(ySigma); 
     
     
     %Compute the goodness of Generalized Gaussian
     dYFit = computeGenGaussianFit(divNormY); % Gen Gaussian fit on DNT of Y 

     %Gen Gaussian fit of DNT of divNormSigY
     yFit = computeGenGaussianFit(divNormSigY);
     
     feat = [yFit dYFit];
 end
