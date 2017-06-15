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
%

% Reshape Y
new_y = zeros(size(y,1),num_labels);
for i = 1: size(y,1)
  new_y(i, y(i)) = 1;
end

% Add bias unit to a1, aka X
X = [ones(m,1), X];
Z2 = X*Theta1';
% a2(i,j) represents the jth unit for the ith training example
% e.g., the first row represents the hidden layer for the first training ex
a2 = sigmoid(Z2);

% Add bias unit to a2
a2 = [ones(size(a2,1),1), a2];
Z3 = a2*Theta2';
a3 = sigmoid(Z3);

% Recall that J = (1/m) SUM(1:m) SUM(1:K) -y_k_i*log(h_theta(x))- (1-y_k_i)*log(1-h_theta(x))
% Where y is the given outputs, and h_theta(x) is a3
J = (-new_y .* log(a3)) - ((1-new_y) .* log(1-a3));
J = (1/m)*sum(sum(J));

% Throw in Regularization - essentially sum the squares of all Weight matrices
regular_sum = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
J = J + regular_sum;

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
%

% delta_3(i,j) represents the error in the jth unit of the ith training example
delta_3 = a3 - new_y;

% To backpropagate the error, multiply an error by the weights used to generate it
% delta_2(i,j) represents the error in the jth unit of the ith training example
delta_2 = (delta_3 * Theta2);
delta_2(:,1) = [];
delta_2 = delta_2 .* sigmoidGradient(Z2);

% REMINDER: We desire the matrices dTheta1 and dTheta2, which represent the 
% derivative of each weight in matrices Theta1 and Theta2
% Consider the case where a2 and delta_3 are both column vectors
% a2 contains m elements, same as the number of units in layer 2
% delta_3 contains n elements, same as the number of units in layer 3
% Theta2 contains m by n elements
% Hence a2 * delta_3' 

% Theta2 is essentially an m by n matrix_type
% Where the ith row, represents delta_3(i) * a2
% In other words, Theta2(i,j) represents delta_3(i) * a2(j)

% Derive Dtheta2
Dtheta2 = zeros(size(Theta2));
for i=1:size(delta_3,1)
  curr_delta = delta_3(i,:);
  curr_a2 = a2(i,:);
  temp = curr_delta' * curr_a2;
 
  Dtheta2 += temp; 
end

Dtheta2 = Dtheta2./m;
Theta2_grad = Dtheta2;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda/m) .* Theta2(:,2:end));

% Derive Dtheta1
Dtheta1 = zeros(size(Theta1));
for i=1:size(delta_2,1)
  curr_delta = delta_2(i,:);
  curr_a1 = X(i,:);
  temp = curr_delta' * curr_a1;
  Dtheta1 += temp;
end
 
Dtheta1 = Dtheta1./m;
Theta1_grad = Dtheta1;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda/m) .* Theta1(:,2:end));

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll the gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
