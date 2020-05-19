%{
Author: Deepti Ghadiyaram

Description: Given an input RGB image, this method transforms the input
into LMS color space.
%}

function lms= convertRGBToLMS(rgb)
    lms = (colorspace('RGB->CAT02 LMS',rgb));
end
