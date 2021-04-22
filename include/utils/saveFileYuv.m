function saveFileYuv(mov, fileName, mode)
% save RGB movie [0, 255] to YUV 4:2:0 file

switch mode
	case 1 % replace file
		fileId = fopen(fileName, 'w');
	case 2 % append to file
		fileId = fopen(fileName, 'a');
	otherwise
		fileId = fopen(fileName, 'w');
end

dim = size(mov(1).cdata);
nrFrame = length(mov);

for f = 1 : 1 : nrFrame
	imgRgb = frame2im(mov(f));
	
	% convert YUV to RGB
	imgYuv = reshape(convertRgbToYuv(reshape(imgRgb, dim(1) * dim(2), 3)), dim(1), dim(2), 3);

	% write Y component
	buf = reshape(imgYuv(:, :, 1).', [], 1); % reshape
	count = fwrite(fileId, buf, 'uchar');
			
	% write U component
	buf = reshape(imgYuv(1 : 2 : end, 1 : 2 : end, 2).', [], 1); % downsample and reshape
	count = fwrite(fileId, buf, 'uchar');

	% write V component
	buf = reshape(imgYuv(1 : 2 : end, 1 : 2 : end, 3).', [], 1); % downsample and reshape
	count = fwrite(fileId, buf, 'uchar');
end

fclose(fileId);

