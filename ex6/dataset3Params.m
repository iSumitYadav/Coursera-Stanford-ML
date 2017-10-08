function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

res = eye(64, 3);
errRow = 0;

for C = [0.01 0.03 0.1 0.3 1 3 10 30]
	for sigma = [0.01 0.03 0.1 0.3 1 3 10 30]
		model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		predictions = svmPredict(model, Xval);
		predictionError = mean(double(predictions ~= yval));

		% store C in col 1, sigma in col 2, and predictionError in 3
		errRow += 1;
		res(errRow, :) = [C, sigma, predictionError];
	end
end

% Final result is sorted based on min error first
sortedResult = sortrows(res, 3);

% Return the first row C and sigma as they are having the min error
C = sortedResult(1, 1);
sigma = sortedResult(1, 2);

% =========================================================================

end
