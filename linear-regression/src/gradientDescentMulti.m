function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
dtheta = zeros(length(theta),1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    temp = X*theta;
    temp = temp - y;
    
    for i = 1:length(theta)  
      dtheta(i) = (1/m)*sum(temp .* X(:,i));
    end
    % dtheta_n = (1/m) * SUM i:m (htheta(X) - y) * X
    
    % Each row of X*theta is equal to h_theta(X_m), the hypothesis with
    % the mth training example as input
    
    
    % Assuming that dtheta_n is a column vector of length m, where
    % the mth row is [ htheta(X_m) - y_m ], where X_m is the vector with values
    % from the mth training example. We should multiply dtheta_n element-wise
    % with X(:, n)
    %
    % Then dJ/dtheta_n = sum(dtheta_n)
    
    theta = theta - alpha * dtheta;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
