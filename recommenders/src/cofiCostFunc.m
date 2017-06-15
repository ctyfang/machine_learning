function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);
                
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% STEP 1 - Compute the (regularized) cost
discrep = R .* (X*Theta' - Y);
predictions = discrep .^2;
J = 0.5*sum(sum(predictions));
X_sum = sum((X(:)).^2);
Theta_sum = sum((Theta(:)).^2);
J = J + (lambda/2)*(Theta_sum + X_sum);

% STEP 2 - Compute the gradients for Theta and X
% Recall that X_grad and Theta_grad are matrices of the derivative at the
% current parameters

for i = 1:num_movies
  for j = 1:num_features
    X_grad(i,j) = sum(discrep(i,:) .* Theta(:,j)');
  end
end

for i = 1:num_users
  for j = 1:num_features
    Theta_grad(i,j) = sum(discrep(:,i) .* X(:,j));
  end
end

X_grad = X_grad + (lambda .* X);
Theta_grad = Theta_grad + (lambda .* Theta);

%if (size(X) == size(X_grad))
%  printf("X_grad correct dim")
%endif

%if (size(Theta) == size(Theta_grad))
%  printf("Theta_grad correct dim")
%endif

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
