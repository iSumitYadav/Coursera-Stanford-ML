function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

const_val = 1/m;
%Add bias unit
a1 = [ones(m, 1) X];

z2 = a1*Theta1';
a2 = sigmoid(z2);
%Add bias unit
size_a2 = size(a2,1);
a2 = [ones(size_a2, 1) a2];

z3 = a2*Theta2';
h_of_theta = sigmoid(z3); %As h_of_theta is equal to a3

yVector = zeros(m, num_labels);

for i = 1:m
	yVector(i, y(i)) = 1;
end

J = const_val*sum(sum(-yVector.*log(h_of_theta) - (1 - yVector).*log(1-h_of_theta)));

regularization = (const_val*lambda/2) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

J = J + regularization;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for i = 1:m
	% for l = 1 i.e. input layer
	a1 = [1; X(i,:)'];

	% for l = 2 i.e. second(hidden) layer
	z2 = Theta1*a1;
	a2 = [1; sigmoid(z2)];

	% for l = 3 i.e. third(output) layer
	z3 = Theta2*a2;
	a3 = sigmoid(z3);

	yk = ([1:num_labels] == y(i))';

	delta3 = a3 - yk;

	delta2 = (Theta2'*delta3).*[1; sigmoidGradient(z2)];
	delta2 = delta2(2:end); % bias removed

	% No need for delta1, as it'll be for input which isn't logical to calculate error in input

	% calculate Caps delta
	Theta1_grad += delta2*a1';
	Theta2_grad += delta3*a2';
end

Theta1_grad = const_val * (Theta1_grad + lambda*[zeros(size(Theta1, 1), 1) Theta1(:,2:end)]);
Theta2_grad = const_val * (Theta2_grad + lambda*[zeros(size(Theta2, 1), 1) Theta2(:,2:end)]);

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
