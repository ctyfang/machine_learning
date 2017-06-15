function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Linear regression assumes that Y is linearly related to all input features
% In other words, the vector theta is one dimensional, of some length related
% to the number of input features
% Recall that the cost is [h(X)-y]^2 + Regularization term

% STEP 1 - Add bias unit to X
% X = [ones(m, 1) X];

% STEP 2 - Calculate h(X)
h = theta' * X';
h = h';

% STEP 3 - Calculate error term
err_sq = (1 / (2*m)) * sum((h - y).^2);

% STEP 4 - Calculate regularization term
err_reg = (lambda / (2*m)) * sum( theta(2:end).^2 );

% STEP 5 - Calculate error
J = err_sq + err_reg;

% Recall that the gradient is dJ/dtheta_i
% To calculate dJ/dtheta_i for each i, we sum the errors from each
% training example.

% Recall that the equations for dJ/dtheta_0 and dJ/dtheta_i where i =/= 0,
% differ due to the regularization term (Theta_0 is not regularized, as it
% is the intercept). 

% STEP 1 - calculate h(X) - y
grad_diff = (1/m) .* (h - y);

% STEP 2 - calculate dJ/dtheta_0
grad(1) = sum(grad_diff .* X(:,1));

% STEP 3 - calculate dJ/dtheta_i, where i is non-zeros
for i = 2:size(theta)
  grad(i) = sum(grad_diff .* X(:,i));
  grad(i) += (lambda/m) * theta(i);
endfor

% =========================================================================

grad = grad(:);

end
