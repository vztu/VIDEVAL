%{
Author: Deepti Ghadiyaram

Description: Given an input RGB image, this method transforms the input
into HSI color space.
%}

function hsi= convertRGBToHSI(rgb)
    hsi = (colorspace('RGB->HSI',rgb));
end
