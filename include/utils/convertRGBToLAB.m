%{
Author: Deepti Ghadiyaram

Description: Given an input RGB image, this method transforms the input
into LAB color space.
%}

function lab = convertRGBToLAB(I)
    gfrgb = imfilter(I, fspecial('gaussian', 3, 3), 'symmetric', 'conv');
    
    %---------------------------------------------------------
    % Perform sRGB to CIE Lab color space conversion (using D65)
    %---------------------------------------------------------
    cform = makecform('srgb2lab', 'AdaptedWhitePoint', whitepoint('D65'));

    lab = applycform(gfrgb,cform);
end