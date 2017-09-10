function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

const_val = 1/m;
h_of_theta = sigmoid(X*theta);

J = const_val*(sum(-y.*log(h_of_theta) - (1-y).*log(1-h_of_theta)) + (lambda/2)*sum(theta(2:end).^2));

%theta0 shouldn't be regularized
grad(1) = const_val*sum((h_of_theta - y).*X(:,1));

% for loop start from 2 coz in octave indexing start from 1 and we don't want to regularize theta0 or in octave's case 1
for i=2:size(theta, 1)
	grad(i) = const_val*(sum((h_of_theta - y).*X(:,i)) + lambda*theta(i));
end


% =============================================================

end
