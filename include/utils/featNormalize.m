function [trainX, testX] = featNormalize(trainX, testX)
    
	minimums = min(trainX, [], 1);
	ranges = max(trainX, [], 1) - minimums;

	trainX = (trainX - repmat(minimums, size(trainX, 1), 1)) ./ repmat(ranges, size(trainX, 1), 1);

	testX = (testX - repmat(minimums, size(testX, 1), 1)) ./ repmat(ranges, size(testX, 1), 1);
	
end
