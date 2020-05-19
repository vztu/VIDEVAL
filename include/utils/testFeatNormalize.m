%% This method normalizes the features to fall in the range [-1, 1]

function testX = testFeatNormalize(testX, minimums, ranges)
   
	testX = (testX - repmat(minimums, size(testX, 1), 1)) ./ repmat(ranges, size(testX, 1), 1);
	
end
