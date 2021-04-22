function yuv = convertRgbToYuv(rgb)
% convert row vector RGB [0, 255] to row vector YUV [0, 255]

load conversion.mat;

rgb = double(rgb);

yuv = (rgbToYuv * rgb.').';
yuv(:, 2 : 3) = yuv(:, 2 : 3) + 127;

yuv = uint8(clipValue(yuv, 0, 255));